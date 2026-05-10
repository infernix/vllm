#!/bin/bash
set -euo pipefail


PORT="${PORT:-8081}"
MODEL="${MODEL:-/home/gerben/ai/models/DeepSeek-V4-Flash}"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"
export CUDA_ARCH_LIST="${CUDA_ARCH_LIST:-120a}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0a}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-100000}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP="${VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP:-1}"
export VLLM_ENABLE_DEEPSEEK_V4_MHC_WARMUP="${VLLM_ENABLE_DEEPSEEK_V4_MHC_WARMUP:-1}"

exec env -u PYTORCH_CUDA_ALLOC_CONF \
  /home/gerben/ai/vllm/.venv/bin/vllm serve "$MODEL" \
  --served-model-name DeepSeek-V4-Flash \
  --host 127.0.0.1 \
  --port "$PORT" \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.99 \
  --tensor-parallel-size 2 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  "$@"
