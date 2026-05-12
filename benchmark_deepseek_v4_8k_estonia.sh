#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$ROOT/run_deepseek_v4_flash_sm120.sh"
BENCH_PYTHON="${BENCH_PYTHON:-/home/gerben/ai/llm-inference-bench/.venv/bin/python}"
BENCH_SCRIPT="${BENCH_SCRIPT:-/home/gerben/ai/llm-inference-bench/llm_decode_bench.py}"
PORT="${PORT:-8085}"
SESSION="${TMUX_SESSION:-deepseek-quality-bench}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"
PROFILE_RUNS="${PROFILE_RUNS:-30}"
PROFILE_CONCURRENCY="${PROFILE_CONCURRENCY:-0}"
PROFILE_MIN_RESULTS="${PROFILE_MIN_RESULTS:-$PROFILE_RUNS}"
RUN_DIR="${RUN_DIR:-/tmp/deepseek-v4-quality-bench}"
SERVER_LOG="$RUN_DIR/server.log"
DECODE_LOG="$RUN_DIR/decode-8k.log"
DECODE_JSON="$RUN_DIR/decode-8k.json"
PROFILE_LOG="$RUN_DIR/profile-estonia.log"
PROFILE_JSON="$RUN_DIR/profile-estonia.json"
HEALTH_URL="http://127.0.0.1:${PORT}/health"

cleanup() {
  if [[ "${KEEP_TMUX_SESSION:-0}" != "1" && "${REUSE_SERVER:-0}" != "1" ]]; then
    tmux kill-session -t "$SESSION" 2>/dev/null || true
  fi
}
trap cleanup EXIT

[[ -x "$LAUNCHER" ]] || { echo "launcher missing or not executable: $LAUNCHER" >&2; exit 1; }
[[ -x "$BENCH_PYTHON" ]] || { echo "benchmark python missing: $BENCH_PYTHON" >&2; exit 1; }
[[ -f "$BENCH_SCRIPT" ]] || { echo "benchmark script missing: $BENCH_SCRIPT" >&2; exit 1; }
mkdir -p "$RUN_DIR"
rm -f "$DECODE_LOG" "$DECODE_JSON" "$PROFILE_LOG" "$PROFILE_JSON"

if [[ "${REUSE_SERVER:-0}" != "1" ]]; then
  rm -f "$SERVER_LOG"
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  pkill -f '/home/gerben/ai/vllm/.venv/bin/vllm serve' || true
  sleep 2
  tmux new-session -d -s "$SESSION" "cd \"$ROOT\" && PORT=\"$PORT\" \"$LAUNCHER\" >\"$SERVER_LOG\" 2>&1"
fi

start_time=$(date +%s)
while true; do
  if curl -sf --max-time 5 "$HEALTH_URL" >/dev/null; then
    break
  fi
  if [[ "${REUSE_SERVER:-0}" != "1" ]] && ! tmux has-session -t "$SESSION" 2>/dev/null; then
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
  --output "$DECODE_JSON" \
  >"$DECODE_LOG" 2>&1

profile_args=(
  --port "$PORT"
  --model DeepSeek-V4-Flash
  --profile estonia
  --completion-stats-runs "$PROFILE_RUNS"
  --completion-stats-min-results "$PROFILE_MIN_RESULTS"
  --display-mode live
  --output "$PROFILE_JSON"
)
if (( PROFILE_CONCURRENCY > 0 )); then
  profile_args+=(--completion-stats-concurrency "$PROFILE_CONCURRENCY")
fi

"$BENCH_PYTHON" "$BENCH_SCRIPT" "${profile_args[@]}" >"$PROFILE_LOG" 2>&1

"$ROOT/.venv/bin/python" - "$DECODE_JSON" "$PROFILE_JSON" <<'PY'
import json
import sys
from pathlib import Path

decode = json.loads(Path(sys.argv[1]).read_text())
profile = json.loads(Path(sys.argv[2]).read_text())

prefill = decode.get("prefill", {})
if not prefill:
    raise SystemExit("missing prefill results")
ctx_key = sorted(prefill.keys(), key=lambda x: int(x))[0]
prefill_s = float(prefill[ctx_key]["tok_per_sec"])
summary = decode.get("summary_table", {})
if ctx_key not in summary:
    raise SystemExit(f"missing decode summary for context {ctx_key}")
conc_key = sorted(summary[ctx_key].keys(), key=lambda x: int(x))[0]
token_gen_s = float(summary[ctx_key][conc_key])

selected = profile.get("selected_summary") or profile.get("all_summary") or {}
if "correct_rate" not in selected:
    raise SystemExit("missing estonia correctness rate")
correctness_rate = float(selected["correct_rate"])
correct = int(selected.get("correct", 0))
completed = int(selected.get("completed", 0))

print(
    f"METRIC prefill_s={prefill_s:g} "
    f"token_gen_s={token_gen_s:g} "
    f"estonia_correctness_rate={correctness_rate:g} "
    f"estonia_correct={correct}/{completed}"
)
PY
