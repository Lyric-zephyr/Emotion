#!/usr/bin/env bash
# 创建项目虚拟环境并安装依赖。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "[venv] 创建: $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "[venv] 已存在: $VENV_DIR"
fi

PYTHON="$VENV_DIR/bin/python"
echo "[venv] 升级 pip 并安装 requirements.txt"
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r "$PROJECT_DIR/requirements.txt"
echo "[venv] 完成。激活命令: source \"$VENV_DIR/bin/activate\""
