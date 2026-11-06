#!/usr/bin/env bash
# 启动 Gradio 推理网页；访问 http://localhost:7860。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$SCRIPT_DIR/setup_venv.sh"
fi

cd "$PROJECT_DIR"
exec "$VENV_DIR/bin/python" app.py "$@"
