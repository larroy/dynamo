#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Tear down the Inkling recipe deployment.
#
# Usage:
#   export NAMESPACE=your-namespace
#   bash cleanup.sh [--delete-pvc] [--delete-namespace]
#
# By default the 592 GB model PVC and the namespace are preserved so that
# a redeploy can reuse cached weights without re-downloading. Pass
# --delete-pvc and/or --delete-namespace to remove them.

set -euo pipefail

DELETE_PVC=0
DELETE_NAMESPACE=0

for arg in "$@"; do
  case "$arg" in
    --delete-pvc)       DELETE_PVC=1 ;;
    --delete-namespace) DELETE_NAMESPACE=1 ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: $0 [--delete-pvc] [--delete-namespace]"
      exit 1
      ;;
  esac
done

: "${NAMESPACE:?Set NAMESPACE before running this script (export NAMESPACE=...)}"

echo "==> Namespace: ${NAMESPACE}"

# Stop any local port-forward for the frontend service
echo "==> Stopping port-forward (if running)..."
pkill -f "kubectl port-forward svc/tml-inkling-sglang-agg-frontend" 2>/dev/null || true

# Delete the DynamoGraphDeployment (cascades to pods and services)
echo "==> Deleting DynamoGraphDeployment tml-inkling-sglang-agg..."
kubectl delete dynamographdeployment tml-inkling-sglang-agg \
  -n "${NAMESPACE}" --ignore-not-found=true

# Delete the model-download job
echo "==> Deleting model-download job..."
kubectl delete job inkling-model-download \
  -n "${NAMESPACE}" --ignore-not-found=true

# Delete secrets
echo "==> Deleting secrets (nvcr-imagepullsecret, hf-token-secret)..."
kubectl delete secret nvcr-imagepullsecret hf-token-secret \
  -n "${NAMESPACE}" --ignore-not-found=true

if [[ "${DELETE_PVC}" -eq 1 ]]; then
  echo "==> Deleting PVC model-cache (WARNING: destroys 592 GB of downloaded weights)..."
  kubectl delete pvc model-cache -n "${NAMESPACE}" --ignore-not-found=true
else
  echo "==> Skipping PVC deletion (pass --delete-pvc to remove the 592 GB model cache)"
fi

if [[ "${DELETE_NAMESPACE}" -eq 1 ]]; then
  echo "==> Deleting namespace ${NAMESPACE} (WARNING: destroys all resources in it)..."
  kubectl delete namespace "${NAMESPACE}" --ignore-not-found=true
else
  echo "==> Skipping namespace deletion (pass --delete-namespace to remove it)"
fi

echo "==> Cleanup complete."
