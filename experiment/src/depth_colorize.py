"""深度图 TIFF 转 Jet 伪彩色可视化工具"""

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def load_depth_tiff(tiff_path: Path) -> np.ndarray:
    """加载 TIFF 格式深度图"""
    img = Image.open(tiff_path)
    depth = np.array(img, dtype=np.float32)
    return depth


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    """将深度值归一化到 [0, 1] 范围"""
    valid_mask = np.isfinite(depth) & (depth > 0)
    if not valid_mask.any():
        return np.zeros_like(depth)

    min_val = depth[valid_mask].min()
    max_val = depth[valid_mask].max()

    if max_val - min_val < 1e-6:
        return np.zeros_like(depth)

    normalized = (depth - min_val) / (max_val - min_val)
    normalized[~valid_mask] = 0
    return normalized


def create_jet_colormap() -> np.ndarray:
    """
    创建 Jet 色彩映射表 (256 级)
    颜色过渡: 深蓝 → 蓝 → 青 → 绿 → 黄 → 橙 → 红
    """
    lut = np.zeros((256, 3), dtype=np.uint8)

    for i in range(256):
        t = i / 255.0

        # Jet colormap 分段线性插值
        if t < 0.125:
            r, g, b = 0, 0, 0.5 + t * 4
        elif t < 0.375:
            r, g, b = 0, (t - 0.125) * 4, 1
        elif t < 0.625:
            r, g, b = (t - 0.375) * 4, 1, 1 - (t - 0.375) * 4
        elif t < 0.875:
            r, g, b = 1, 1 - (t - 0.625) * 4, 0
        else:
            r, g, b = 1 - (t - 0.875) * 2, 0, 0

        lut[i] = [
            int(min(255, max(0, r * 255))),
            int(min(255, max(0, g * 255))),
            int(min(255, max(0, b * 255)))
        ]

    return lut


def apply_colormap(
    normalized_depth: np.ndarray,
    colormap: np.ndarray
) -> np.ndarray:
    """应用伪彩色映射"""
    indices = (normalized_depth * 255).astype(np.uint8)
    height, width = normalized_depth.shape
    colored = colormap[indices.flatten()].reshape(height, width, 3)
    return colored


def depth_to_pseudocolor(
    depth: np.ndarray,
    invert: bool = True
) -> np.ndarray:
    """
    将深度图转换为 Jet 伪彩色图像

    Args:
        depth: 深度数组
        invert: 是否反转 (True: 近处为红色, 远处为蓝色)

    Returns:
        RGB 伪彩色图像数组
    """
    normalized = normalize_depth(depth)

    if invert:
        normalized = 1.0 - normalized

    colormap = create_jet_colormap()
    colored = apply_colormap(normalized, colormap)

    return colored


def process_single_file(
    input_path: Path,
    output_path: Path,
    invert: bool = True
) -> None:
    """处理单个深度图文件"""
    depth = load_depth_tiff(input_path)
    colored = depth_to_pseudocolor(depth, invert=invert)

    img = Image.fromarray(colored, mode='RGB')
    img.save(output_path, quality=95)
    print(f"已保存: {output_path}")


def process_directory(
    input_dir: Path,
    output_dir: Path,
    invert: bool = True
) -> Tuple[int, int]:
    """
    批量处理目录中的所有 TIFF 深度图

    Returns:
        (成功数量, 失败数量)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    tiff_files = list(input_dir.glob("*.tiff")) + list(input_dir.glob("*.tif"))
    tiff_files = sorted(set(tiff_files))

    if not tiff_files:
        print(f"警告: 在 {input_dir} 中未找到 TIFF 文件")
        return 0, 0

    success_count = 0
    fail_count = 0

    for tiff_path in tiff_files:
        output_name = tiff_path.stem + ".png"
        output_path = output_dir / output_name

        try:
            process_single_file(tiff_path, output_path, invert=invert)
            success_count += 1
        except Exception as e:
            print(f"处理失败 {tiff_path.name}: {e}")
            fail_count += 1

    return success_count, fail_count


def main() -> None:
    """主函数"""
    project_root = Path(__file__).parent.parent
    input_dir = project_root / "depth_input"
    output_dir = project_root / "depth_output"

    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print("-" * 40)

    success, fail = process_directory(input_dir, output_dir)

    print("-" * 40)
    print(f"处理完成: 成功 {success} 个, 失败 {fail} 个")


if __name__ == "__main__":
    main()
