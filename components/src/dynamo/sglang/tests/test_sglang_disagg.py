# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SGLang unified engine's disagg dispatch."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("sglang", reason="sglang not installed in this container")

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST  # noqa: E402

from dynamo.common.constants import DisaggregationMode  # noqa: E402
from dynamo.sglang._disagg import warmup_prefill_engine  # noqa: E402
from dynamo.sglang.llm_engine import SglangLLMEngine  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.unified,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _engine_with_cached_bootstrap() -> SglangLLMEngine:
    # __new__ bypasses real engine init; we only exercise pure-Python helpers.
    engine = SglangLLMEngine.__new__(SglangLLMEngine)
    engine._bootstrap_host = "10.0.0.5"
    engine._bootstrap_port = 12345
    engine.serving_mode = DisaggregationMode.PREFILL
    return engine


def test_prefill_bootstrap_honours_router_provided_triple():
    # Router-resolved triple must be passed through unchanged so the
    # decode peer (which the router writes the same triple onto) finds
    # the matching room.
    engine = _engine_with_cached_bootstrap()
    out = engine._resolve_prefill_bootstrap(
        {
            "token_ids": [1, 2, 3],
            "bootstrap_info": {
                "bootstrap_host": "router.host",
                "bootstrap_port": 9999,
                "bootstrap_room": 42,
            },
        }
    )
    assert out == {
        "bootstrap_host": "router.host",
        "bootstrap_port": 9999,
        "bootstrap_room": 42,
    }


def test_decode_bootstrap_raises_when_router_left_no_info():
    # Both router paths missing → loud failure. Silent zero-fill would
    # leave the decode worker waiting on a phantom KV transfer.
    with pytest.raises(ValueError, match="bootstrap"):
        SglangLLMEngine._resolve_decode_bootstrap({"token_ids": [1, 2, 3]})


@pytest.mark.asyncio
async def test_prefill_warmup_covers_every_dp_rank():
    calls = []
    consumed_dp_ranks = []

    async def async_generate(**kwargs):
        calls.append(kwargs)

        async def results():
            dp_rank = kwargs["routed_dp_rank"]
            consumed_dp_ranks.append(dp_rank)
            yield {"meta_info": {"dp_rank": dp_rank}}

        return results()

    engine = SimpleNamespace(
        async_generate=async_generate,
        tokenizer_manager=SimpleNamespace(server_args=SimpleNamespace(dp_size=8)),
    )

    await warmup_prefill_engine(engine, bootstrap_port=8998)

    assert len(calls) == 8
    calls.sort(key=lambda call: call["routed_dp_rank"])
    for dp_rank, call in enumerate(calls):
        assert call["input_ids"] == [10, 11, 12, 13]
        assert call["bootstrap_host"] == FAKE_BOOTSTRAP_HOST
        assert call["bootstrap_port"] == 8998
        assert call["bootstrap_room"] == dp_rank
        assert call["routed_dp_rank"] == dp_rank
    assert sorted(consumed_dp_ranks) == list(range(8))
