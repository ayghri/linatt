#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Pre-tokenize FineWeb-Edu sample-10BT. Cache lives at data/.
python prepare.py "$@"
