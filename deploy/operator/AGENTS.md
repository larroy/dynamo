<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

- RBAC changes for operator code must update the `+kubebuilder:rbac` markers.
- Run `make manifests` to regenerate both `config/rbac/role.yaml` and the
  platform chart's `../helm/charts/platform/components/operator/files/manager-role.yaml`.
- Keep chart-only grants in the manual section of the platform chart's
  `../helm/charts/platform/components/operator/templates/manager-rbac.yaml`.

## Go Test Style

- Use `t.Log` to tell the test story. Prefer one `t.Log` heading before each
  block of code that implements a test step.
- Avoid test construction where the bespoke test behavior lives inside a
  closure. Keep the test flow linear. Closures are acceptable only for a very
  small number of standardized core helpers such as `Eventually`-style waits.
- Avoid shared test fixtures such as reusable DGD or DGDR objects. Each test
  should own the object fixtures it creates.
- Keep test fixtures as local variables or constants. This makes fixture
  ownership obvious and keeps tests independent.
- When fixture construction is complex and repeats boilerplate, prefer a small
  object-construction DSL over shared fixture snippets. The DSL should describe
  what is constructed at a high level; the resulting fixture still belongs to
  the individual test.
