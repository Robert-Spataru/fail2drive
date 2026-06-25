#!/usr/bin/env bash
# Pre-download Qwen3-VL-8B-Instruct into the shared HF cache.
set -euo pipefail

HF_HOME_DIR="${HF_HOME_DIR:-/data/robert/models/hf}"
HF_HUB_CACHE_DIR="${HF_HUB_CACHE_DIR:-${HF_HOME_DIR}/hub}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
PYTHON_BIN="${PYTHON_BIN:-/data/robert/fail2drive/env/bin/python}"

mkdir -p "${HF_HUB_CACHE_DIR}"

HF_HOME="${HF_HOME_DIR}" \
HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE_DIR}" \
MODEL_ID="${MODEL_ID}" \
"${PYTHON_BIN}" - <<'PY'
import os
from huggingface_hub import snapshot_download

model_id = os.environ.get("MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")
cache_dir = os.environ["HUGGINGFACE_HUB_CACHE"]
path = snapshot_download(repo_id=model_id, cache_dir=cache_dir)
print(f"Downloaded {model_id} -> {path}")
PY
