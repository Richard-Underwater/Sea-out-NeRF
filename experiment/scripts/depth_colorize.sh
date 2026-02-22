#!/bin/bash

# 深度图伪彩色转换脚本
# 将 depth_input 中的 TIFF 文件转换为红黄伪彩色图保存到 depth_output

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT" || exit 1

uv run python -m src.depth_colorize
