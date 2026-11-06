#!/usr/bin/env bash
# 使用项目虚拟环境启动训练；所有参数原样传给 train.py。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$SCRIPT_DIR/setup_venv.sh"
fi

cd "$PROJECT_DIR"
exec "$VENV_DIR/bin/python" train.py "$@"
