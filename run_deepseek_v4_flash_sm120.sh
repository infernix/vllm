#!/bin/bash
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"
export CUDA_ARCH_LIST="${CUDA_ARCH_LIST:-120a}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0a}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-100000}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP="${VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP:-0}"
export VLLM_ENABLE_DEEPSEEK_V4_MHC_WARMUP="${VLLM_ENABLE_DEEPSEEK_V4_MHC_WARMUP:-0}"
unset PYTORCH_CUDA_ALLOC_CONF || true

PORT="${PORT:-8081}"
MODEL="${MODEL:-/home/gerben/ai/models/DeepSeek-V4-Flash}"

exec ./.venv/bin/vllm serve "$MODEL" \
  --served-model-name DeepSeek-V4-Flash \
  --host 127.0.0.1 \
  --port "$PORT" \
  --trust-remote-code \
  --disable-custom-all-reduce \
  --no-async-scheduling \
  --enforce-eager \
  --kv-cache-dtype fp8_e4m3 \
  --block-size 256 \
  --max-model-len 500000 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 2 \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  "$@"
