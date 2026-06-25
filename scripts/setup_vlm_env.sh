#!/usr/bin/env bash
# Create a dedicated VLM Python env for Qwen3-VL subprocess inference.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VLM_ENV_DIR="${VLM_ENV_DIR:-${ROOT}/env_vlm}"
export HF_HOME_DIR="${HF_HOME_DIR:-/data/robert/models/hf}"
export HF_HUB_CACHE_DIR="${HF_HUB_CACHE_DIR:-${HF_HOME_DIR}/hub}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"

bash /data/robert/AutoAgent0/scripts/setup_vlm_env.sh

echo ""
echo "VLM env ready: ${VLM_ENV_DIR}/bin/python"
echo "Set vlm.python_bin in your YAML to: ${VLM_ENV_DIR}/bin/python"
