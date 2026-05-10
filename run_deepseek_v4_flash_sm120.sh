#!/bin/bash
set -euo pipefail


PORT="${PORT:-8081}"
MODEL="${MODEL:-/home/gerben/ai/models/DeepSeek-V4-Flash}"

exec env -u PYTORCH_CUDA_ALLOC_CONF \
  /home/gerben/ai/vllm/.venv/bin/vllm serve "$MODEL" \
  --served-model-name DeepSeek-V4-Flash \
  --host 127.0.0.1 \
  --port "$PORT" \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --max-model-len 500000 \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 2 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  "$@"
