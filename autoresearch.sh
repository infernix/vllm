#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$ROOT/run_deepseek_v4_flash_sm120.sh"
BENCH_PYTHON="/home/gerben/ai/llm-inference-bench/.venv/bin/python"
BENCH_SCRIPT="/home/gerben/ai/llm-inference-bench/llm_decode_bench.py"
PORT="${PORT:-8085}"
SESSION="${TMUX_SESSION:-deepseek-autoresearch}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-180}"
SERVER_LOG="${SERVER_LOG:-/tmp/${SESSION}.server.log}"
BENCH_LOG="${BENCH_LOG:-/tmp/${SESSION}.bench.log}"
BENCH_JSON="${BENCH_JSON:-/tmp/${SESSION}.bench.json}"
HEALTH_URL="http://127.0.0.1:${PORT}/health"

cleanup() {
  tmux kill-session -t "$SESSION" 2>/dev/null || true
}

trap 'if [[ "${KEEP_TMUX_SESSION:-0}" != "1" ]]; then cleanup; fi' EXIT

[[ -x "$LAUNCHER" ]] || { echo "launcher missing or not executable: $LAUNCHER" >&2; exit 1; }
[[ -x "$BENCH_PYTHON" ]] || { echo "benchmark python missing: $BENCH_PYTHON" >&2; exit 1; }
[[ -f "$BENCH_SCRIPT" ]] || { echo "benchmark script missing: $BENCH_SCRIPT" >&2; exit 1; }

rm -f "$SERVER_LOG" "$BENCH_LOG" "$BENCH_JSON"
tmux kill-session -t "$SESSION" 2>/dev/null || true
pkill -f '/home/gerben/ai/vllm/.venv/bin/vllm serve' || true
sleep 2

tmux new-session -d -s "$SESSION" "cd \"$ROOT\" && PORT=\"$PORT\" \"$LAUNCHER\" >\"$SERVER_LOG\" 2>&1"

start_time=$(date +%s)
while true; do
  if curl -sf --max-time 5 "$HEALTH_URL" >/dev/null; then
    break
  fi
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "server exited before health check; inspect $SERVER_LOG" >&2
    exit 1
  fi
  now=$(date +%s)
  if (( now - start_time >= STARTUP_TIMEOUT )); then
    echo "server did not become healthy within ${STARTUP_TIMEOUT}s; inspect $SERVER_LOG" >&2
    exit 1
  fi
  sleep 2
 done

"$BENCH_PYTHON" "$BENCH_SCRIPT" \
  --port "$PORT" \
  --model DeepSeek-V4-Flash \
  --contexts 8k \
  --prefill-contexts 8k \
  --concurrency 1 \
  --max-tokens 8192 \
  --display-mode plain \
  --output "$BENCH_JSON" \
  >"$BENCH_LOG" 2>&1

"/home/gerben/ai/vllm/.venv/bin/python" - "$BENCH_JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
prefill = data.get("prefill", {})
if not prefill:
    raise SystemExit("missing prefill results")
ctx_key = sorted(prefill.keys(), key=lambda x: int(x))[0]
prefill_s = prefill[ctx_key]["tok_per_sec"]
summary = data.get("summary_table", {})
if ctx_key not in summary:
    raise SystemExit(f"missing decode summary for context {ctx_key}")
conc_key = sorted(summary[ctx_key].keys(), key=lambda x: int(x))[0]
token_gen_s = summary[ctx_key][conc_key]
print(f"METRIC prefill_s={prefill_s:g} token_gen_s={token_gen_s:g}")
PY
