#!/bin/bash
# 去水印图像质量评估脚本
# 用法:
#   bash scripts/evaluate_watermark.sh                  # 评估所有方法（含亮度对齐）
#   bash scripts/evaluate_watermark.sh --no-align       # 评估所有方法（不亮度对齐）
#   bash scripts/evaluate_watermark.sh --gen-aligned    # 仅生成亮度对齐后的图像
#   bash scripts/evaluate_watermark.sh -m 方法名        # 评估单个方法

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT" || exit 1

mkdir -p logs

# 运行评估，日志输出到 logs/
uv run python -m src.watermark_eval "$@" 2>&1 | tee logs/watermark_eval.log
