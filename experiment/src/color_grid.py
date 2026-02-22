"""RGB颜色方块拼接工具"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Union
import math

from PIL import Image, ImageDraw


@dataclass
class RGB:
    """RGB颜色"""
    r: int
    g: int
    b: int

    def to_tuple(self) -> tuple[int, int, int]:
        return (self.r, self.g, self.b)

    @classmethod
    def from_hex(cls, hex_str: str) -> "RGB":
        """从十六进制字符串创建，如 '#FF5733' 或 'FF5733'"""
        hex_str = hex_str.lstrip("#")
        return cls(
            r=int(hex_str[0:2], 16),
            g=int(hex_str[2:4], 16),
            b=int(hex_str[4:6], 16),
        )

    @classmethod
    def from_tuple(cls, t: tuple[int, int, int]) -> "RGB":
        """从元组创建"""
        return cls(r=t[0], g=t[1], b=t[2])


@dataclass
class GridConfig:
    """网格配置"""
    cell_size: int = 100  # 每个方块的像素大小
    cols: Optional[int] = None  # 列数，None则自动计算
    border_width: int = 2  # 边框宽度
    border_color: Tuple[int, int, int] = (255, 255, 255)  # 边框颜色


def create_color_grid(
    colors: List[RGB],
    config: Optional[GridConfig] = None,
    output_path: Union[Path, str, None] = None,
) -> Image.Image:
    """
    将多个颜色拼接成网格图片

    Args:
        colors: RGB颜色列表
        config: 网格配置
        output_path: 输出文件路径，None则不保存

    Returns:
        生成的PIL Image对象
    """
    if not colors:
        raise ValueError("颜色列表不能为空")

    cfg = config or GridConfig()
    n = len(colors)

    # 计算网格尺寸
    if cfg.cols:
        cols = cfg.cols
    else:
        cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    # 创建画布
    width = cols * cfg.cell_size
    height = rows * cfg.cell_size
    img = Image.new("RGB", (width, height), cfg.border_color)
    draw = ImageDraw.Draw(img)

    # 绘制每个颜色方块
    for i, color in enumerate(colors):
        row = i // cols
        col = i % cols

        x1 = col * cfg.cell_size + cfg.border_width
        y1 = row * cfg.cell_size + cfg.border_width
        x2 = (col + 1) * cfg.cell_size - cfg.border_width
        y2 = (row + 1) * cfg.cell_size - cfg.border_width

        draw.rectangle([x1, y1, x2, y2], fill=color.to_tuple())

    # 保存图片
    if output_path:
        img.save(str(output_path))

    return img


def parse_color_input(color_input: Union[str, Tuple[int, int, int]]) -> RGB:
    """解析颜色输入，支持多种格式"""
    if isinstance(color_input, tuple):
        return RGB.from_tuple(color_input)

    color_input = color_input.strip()

    # 十六进制格式
    if color_input.startswith("#") or len(color_input) == 6:
        return RGB.from_hex(color_input)

    # rgb(r, g, b) 格式
    if color_input.startswith("rgb"):
        nums = color_input.replace("rgb", "").strip("() ")
        parts = [int(x.strip()) for x in nums.split(",")]
        return RGB(r=parts[0], g=parts[1], b=parts[2])

    # r, g, b 格式
    if "," in color_input:
        parts = [int(x.strip()) for x in color_input.split(",")]
        return RGB(r=parts[0], g=parts[1], b=parts[2])

    raise ValueError(f"无法解析颜色: {color_input}")


def create_grid_from_strings(
    color_strings: List[str],
    output_path: str = "color_grid.png",
    cell_size: int = 100,
    cols: Optional[int] = None,
) -> Path:
    """
    便捷函数：从颜色字符串列表创建网格图片

    Args:
        color_strings: 颜色字符串列表，支持多种格式
        output_path: 输出文件路径
        cell_size: 方块大小
        cols: 列数

    Returns:
        输出文件路径
    """
    colors = [parse_color_input(c) for c in color_strings]
    config = GridConfig(cell_size=cell_size, cols=cols)
    output = Path(output_path)
    create_color_grid(colors, config, output)
    return output
