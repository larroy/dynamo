#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Deploy and smoke-test the Inkling NVFP4 SGLang agg-B200 recipe.

Required environment variables:
  NGC_API_KEY   NGC API key for pulling nvcr.io/nvidia/ai-dynamo images
  HF_TOKEN      HuggingFace token for model download (model is public; token avoids rate limits)

Usage:
  python3 deploy.py [OPTIONS]

Options:
  --namespace NS          Kubernetes namespace  (default: inkling-test)
  --storage-class SC      storageClassName for the model-cache PVC
                          (default: auto-detected from cluster default)
  --model-download-timeout SECS
                          Seconds to wait for model download job  (default: 7200)
  --deploy-timeout SECS   Seconds to wait for DGD Ready  (default: 1800)
  --smoke-only            Skip deploy; only port-forward and smoke test
  --skip-smoke            Deploy but skip the smoke test
  --dry-run               Print kubectl commands without executing them
"""

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

RECIPE_DIR = Path(__file__).parent
MODEL_CACHE_DIR = RECIPE_DIR / "model-cache"
DEPLOY_YAML = RECIPE_DIR / "sglang" / "agg-b200" / "deploy.yaml"
DGD_NAME = "tml-inkling-sglang-agg"
FRONTEND_SVC = f"{DGD_NAME}-frontend"
MODEL_NAME = "thinkingmachines/Inkling-NVFP4"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def step(msg: str) -> None:
    print(f"\n\033[1;34m==> {msg}\033[0m", flush=True)

def ok(msg: str) -> None:
    print(f"    \033[0;32m✓ {msg}\033[0m", flush=True)

def warn(msg: str) -> None:
    print(f"    \033[0;33m⚠ {msg}\033[0m", flush=True)

def die(msg: str) -> None:
    print(f"\n\033[0;31mERROR: {msg}\033[0m", flush=True)
    sys.exit(1)


def run(args: list, *, capture: bool = False, check: bool = True,
        dry_run: bool = False, input_text: str | None = None) -> subprocess.CompletedProcess:
    cmd_str = " ".join(str(a) for a in args)
    if dry_run:
        print(f"    [dry-run] {cmd_str}")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
    result = subprocess.run(
        args,
        capture_output=capture,
        text=True,
        input=input_text,
    )
    if check and result.returncode != 0:
        die(f"Command failed (exit {result.returncode}):\n  {cmd_str}\n{result.stderr or ''}")
    return result


def kubectl(*args, capture: bool = False, check: bool = True,
            dry_run: bool = False, input_text: str | None = None) -> subprocess.CompletedProcess:
    return run(["kubectl", *args], capture=capture, check=check,
               dry_run=dry_run, input_text=input_text)


def kubectl_json(*args) -> dict | list:
    r = kubectl(*args, "-o", "json", capture=True)
    return json.loads(r.stdout)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def preflight(namespace: str, dry_run: bool = False) -> None:
    step("Preflight checks")

    if not shutil.which("kubectl"):
        die("kubectl not found in PATH")
    ok("kubectl found")

    r = kubectl("cluster-info", capture=True, check=False)
    if r.returncode != 0:
        die("kubectl cluster-info failed — is your kubeconfig set?")
    ok("cluster reachable")

    r = kubectl("get", "crd", "dynamographdeployments.nvidia.com",
                capture=True, check=False)
    if r.returncode != 0:
        die(
            "DynamoGraphDeployment CRD not found.\n"
            "Install the Dynamo Platform first:\n"
            "  helm repo add nvidia-dynamo https://helm.ngc.nvidia.com/nvidia/ai-dynamo\n"
            "  helm install dynamo-platform nvidia-dynamo/dynamo-platform \\\n"
            "    --namespace dynamo-system --create-namespace --wait"
        )
    ok("DynamoGraphDeployment CRD present")

    if not dry_run:
        for key in ("NGC_API_KEY", "HF_TOKEN"):
            if not os.environ.get(key, ""):
                die(f"{key} environment variable is not set or empty")
        ok("NGC_API_KEY and HF_TOKEN set")
    else:
        warn("dry-run: skipping credential check")


def detect_storage_class() -> str:
    sc_list = kubectl_json("get", "storageclass")
    for sc in sc_list.get("items", []):
        annotations = sc.get("metadata", {}).get("annotations", {})
        if annotations.get("storageclass.kubernetes.io/is-default-class") == "true":
            return sc["metadata"]["name"]
    die(
        "No default StorageClass found — pass --storage-class SC to specify one.\n"
        "  kubectl get storageclass"
    )


# ---------------------------------------------------------------------------
# Namespace + secrets
# ---------------------------------------------------------------------------

def ensure_namespace(namespace: str, dry_run: bool) -> None:
    step(f"Namespace: {namespace}")
    r = kubectl("get", "namespace", namespace, capture=True, check=False)
    if r.returncode == 0:
        ok(f"namespace/{namespace} already exists")
    else:
        kubectl("create", "namespace", namespace, dry_run=dry_run)
        ok(f"namespace/{namespace} created")


def ensure_secret(namespace: str, name: str, dry_run: bool, **kwargs) -> None:
    r = kubectl("get", "secret", name, "-n", namespace, capture=True, check=False)
    if r.returncode == 0:
        warn(f"secret/{name} already exists — skipping")
        return
    create_args = ["create", "secret", *kwargs.pop("type_args"), name, "-n", namespace]
    for k, v in kwargs.items():
        create_args.append(f"--{k.replace('_', '-')}={v}")
    kubectl(*create_args, dry_run=dry_run)
    ok(f"secret/{name} created")


def create_secrets(namespace: str, dry_run: bool) -> None:
    step("Kubernetes secrets")
    ngc_key = os.environ.get("NGC_API_KEY", "<NGC_API_KEY>")
    hf_token = os.environ.get("HF_TOKEN", "<HF_TOKEN>")

    ensure_secret(
        namespace, "nvcr-imagepullsecret", dry_run,
        type_args=["docker-registry"],
        docker_server="nvcr.io",
        docker_username="$oauthtoken",
        docker_password=ngc_key,
    )
    ensure_secret(
        namespace, "hf-token-secret", dry_run,
        type_args=["generic"],
        **{"from-literal": f"HF_TOKEN={hf_token}"},
    )


# ---------------------------------------------------------------------------
# Model cache PVC
# ---------------------------------------------------------------------------

def ensure_pvc(namespace: str, storage_class: str, dry_run: bool) -> None:
    step("Model-cache PVC")
    r = kubectl("get", "pvc", "model-cache", "-n", namespace, capture=True, check=False)
    if r.returncode == 0:
        ok("PVC model-cache already exists")
        return

    # Patch storageClassName inline
    yaml_text = (MODEL_CACHE_DIR / "model-cache.yaml").read_text()
    import re
    yaml_text = re.sub(
        r'storageClassName:\s*"[^"]*"',
        f'storageClassName: "{storage_class}"',
        yaml_text,
    )

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(yaml_text)
        tmp = f.name

    try:
        kubectl("apply", "-f", tmp, "-n", namespace, dry_run=dry_run)
    finally:
        os.unlink(tmp)

    ok(f"PVC model-cache created (storageClass={storage_class})")


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

def run_model_download(namespace: str, timeout: int, dry_run: bool) -> None:
    step("Model download (thinkingmachines/Inkling-NVFP4, ~592 GB)")

    r = kubectl("get", "job", "inkling-model-download", "-n", namespace,
                capture=True, check=False)
    if r.returncode == 0:
        info = kubectl_json("get", "job", "inkling-model-download", "-n", namespace)
        if info.get("status", {}).get("succeeded", 0) >= 1:
            ok("model-download job already completed")
            return
        warn("model-download job exists but not complete — waiting for it")
    else:
        kubectl("apply", "-f", str(MODEL_CACHE_DIR / "model-download.yaml"),
                "-n", namespace, dry_run=dry_run)
        ok("model-download job submitted")

    if dry_run:
        return

    print(f"    Waiting up to {timeout}s for job completion ...", flush=True)
    kubectl("wait", "--for=condition=Complete", "job/inkling-model-download",
            "-n", namespace, f"--timeout={timeout}s")
    ok("model-download job completed")


# ---------------------------------------------------------------------------
# DynamoGraphDeployment
# ---------------------------------------------------------------------------

def deploy_dgd(namespace: str, dry_run: bool) -> None:
    step("DynamoGraphDeployment")
    kubectl("apply", "-f", str(DEPLOY_YAML), "-n", namespace, dry_run=dry_run)
    ok(f"DGD {DGD_NAME} applied")


def wait_for_dgd(namespace: str, timeout: int, dry_run: bool) -> None:
    if dry_run:
        return

    step(f"Waiting for DGD ready (timeout={timeout}s)")
    deadline = time.monotonic() + timeout
    interval = 15

    while time.monotonic() < deadline:
        r = kubectl("get", "dynamographdeployment", DGD_NAME, "-n", namespace,
                    capture=True, check=False)
        if r.returncode == 0 and "True" in r.stdout:
            ok(f"DGD {DGD_NAME} is Ready")
            return

        # Print pod summary
        pods_r = kubectl("get", "pods", "-n", namespace, capture=True, check=False)
        for line in pods_r.stdout.splitlines():
            if DGD_NAME in line:
                print(f"    {line}", flush=True)

        remaining = int(deadline - time.monotonic())
        if remaining > 0:
            print(f"    Not ready yet — retrying in {interval}s ({remaining}s left) ...",
                  flush=True)
            time.sleep(interval)

    # Print events for diagnosis before dying
    kubectl("get", "events", "-n", namespace, "--sort-by=.lastTimestamp", check=False)
    die(f"DGD {DGD_NAME} did not become Ready within {timeout}s")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def smoke_test(namespace: str) -> None:
    import urllib.request
    import urllib.error

    step("Smoke test")

    port = _free_port()
    pf = subprocess.Popen(
        ["kubectl", "port-forward", f"svc/{FRONTEND_SVC}", f"{port}:8000", "-n", namespace],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://localhost:{port}"

    try:
        # Wait for port-forward to be ready
        for _ in range(20):
            time.sleep(1)
            try:
                urllib.request.urlopen(f"{base}/v1/models", timeout=3)
                break
            except (urllib.error.URLError, ConnectionRefusedError):
                pass
        else:
            die(f"Frontend at {base} did not respond after 20s")

        def chat(payload: dict) -> dict:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{base}/v1/chat/completions",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())

        # /v1/models
        with urllib.request.urlopen(f"{base}/v1/models", timeout=10) as resp:
            models = json.loads(resp.read())
        ids = [m["id"] for m in models.get("data", [])]
        assert MODEL_NAME in ids, f"expected {MODEL_NAME} in /v1/models, got {ids}"
        ok(f"/v1/models: {ids}")

        # Text
        r = chat({
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Hello, who are you? Reply in one sentence."}],
            "max_tokens": 128,
        })
        content = r["choices"][0]["message"]["content"]
        assert content, "text response content is empty"
        ok(f"Text: {content[:80]}")

        # Image
        r = chat({
            "model": MODEL_NAME,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png",
                    }},
                    {"type": "text", "text": "Describe this image in one sentence."},
                ],
            }],
            "max_tokens": 512,
        })
        content = r["choices"][0]["message"]["content"]
        assert content, "image response content is empty"
        ok(f"Image: {content[:80]}")

        # Audio
        r = chat({
            "model": MODEL_NAME,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {
                        "url": "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/mlk.wav",
                    }},
                    {"type": "text", "text": "Transcribe this audio."},
                ],
            }],
            "max_tokens": 256,
        })
        content = r["choices"][0]["message"]["content"]
        assert content, "audio response content is empty"
        ok(f"Audio: {content[:80]}")

        # Reasoning effort
        r = chat({
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "What is 17 times 24?"}],
            "chat_template_kwargs": {"reasoning_effort": "low"},
            "max_tokens": 128,
        })
        content = r["choices"][0]["message"]["content"]
        assert "408" in content, f"expected 408 in answer, got: {content}"
        thinking_chars = len(r["choices"][0]["message"].get("reasoning_content") or "")
        ok(f"Reasoning effort=low: answer='{content.strip()}', thinking_chars={thinking_chars}")

        print(f"\n\033[1;32mAll smoke tests passed.\033[0m")

    finally:
        pf.send_signal(signal.SIGTERM)
        pf.wait(timeout=5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__ or ""),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--namespace", default="inkling-test")
    p.add_argument("--storage-class", default=None,
                   help="storageClassName for model-cache PVC (auto-detected if omitted)")
    p.add_argument("--model-download-timeout", type=int, default=7200,
                   metavar="SECS")
    p.add_argument("--deploy-timeout", type=int, default=1800, metavar="SECS")
    p.add_argument("--smoke-only", action="store_true",
                   help="Skip deploy; only run smoke tests against an existing deployment")
    p.add_argument("--skip-smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Print kubectl commands without executing")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ns = args.namespace
    dry = args.dry_run

    if not args.smoke_only:
        preflight(ns, dry_run=dry)

        storage_class = args.storage_class or (detect_storage_class() if not dry else "standard")
        ok(f"storage class: {storage_class}")

        ensure_namespace(ns, dry)
        create_secrets(ns, dry)
        ensure_pvc(ns, storage_class, dry)
        run_model_download(ns, args.model_download_timeout, dry)
        deploy_dgd(ns, dry)
        wait_for_dgd(ns, args.deploy_timeout, dry)

    if not args.skip_smoke:
        smoke_test(ns)


if __name__ == "__main__":
    main()
