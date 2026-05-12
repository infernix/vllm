#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$ROOT/benchmark_deepseek_v4_8k_estonia.sh"
