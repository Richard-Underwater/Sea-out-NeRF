#!/bin/bash
# 运行颜色网格工具

cd "$(dirname "$0")/.."

# 确保输出目录存在
mkdir -p output

# 安装依赖并运行
uv run python main.py
