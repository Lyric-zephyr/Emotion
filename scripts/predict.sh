#!/usr/bin/env bash
# 启动命令行推理；例如: scripts/predict.sh --ensemble image.jpg
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$SCRIPT_DIR/setup_venv.sh"
fi

cd "$PROJECT_DIR"
exec "$VENV_DIR/bin/python" predict.py "$@"
