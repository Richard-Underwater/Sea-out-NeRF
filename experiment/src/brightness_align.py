"""
亮度对齐模块
将待测图像的亮度调整为参考图像（GT）的亮度
方法：在 LAB 色彩空间中对齐 L 通道均值
"""

import numpy as np
from PIL import Image


def rgb_to_lab(img_rgb: np.ndarray) -> np.ndarray:
    """将 sRGB 图像转换为 CIELAB 色彩空间

    Args:
        img_rgb: uint8 RGB 图像, shape (H, W, 3)

    Returns:
        float64 LAB 图像, shape (H, W, 3)
    """
    # sRGB -> 线性 RGB
    img = img_rgb.astype(np.float64) / 255.0
    mask = img > 0.04045
    img[mask] = ((img[mask] + 0.055) / 1.055) ** 2.4
    img[~mask] = img[~mask] / 12.92

    # 线性 RGB -> XYZ (D65 白点)
    mat = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = img @ mat.T

    # 归一化 (D65 白点)
    xyz[:, :, 0] /= 0.95047
    xyz[:, :, 2] /= 1.08883

    # XYZ -> LAB
    epsilon = 0.008856
    kappa = 903.3
    mask = xyz > epsilon
    f = np.zeros_like(xyz)
    f[mask] = np.cbrt(xyz[mask])
    f[~mask] = (kappa * xyz[~mask] + 16.0) / 116.0

    lab = np.zeros_like(xyz)
    lab[:, :, 0] = 116.0 * f[:, :, 1] - 16.0   # L
    lab[:, :, 1] = 500.0 * (f[:, :, 0] - f[:, :, 1])  # a
    lab[:, :, 2] = 200.0 * (f[:, :, 1] - f[:, :, 2])  # b

    return lab


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """将 CIELAB 图像转换回 sRGB 色彩空间

    Args:
        lab: float64 LAB 图像, shape (H, W, 3)

    Returns:
        uint8 RGB 图像, shape (H, W, 3)
    """
    # LAB -> XYZ
    fy = (lab[:, :, 0] + 16.0) / 116.0
    fx = lab[:, :, 1] / 500.0 + fy
    fz = fy - lab[:, :, 2] / 200.0

    epsilon = 0.008856
    kappa = 903.3

    xr = np.where(fx ** 3 > epsilon, fx ** 3, (116.0 * fx - 16.0) / kappa)
    yr = np.where(lab[:, :, 0] > kappa * epsilon,
                  fy ** 3, lab[:, :, 0] / kappa)
    zr = np.where(fz ** 3 > epsilon, fz ** 3, (116.0 * fz - 16.0) / kappa)

    xyz = np.stack([xr * 0.95047, yr, zr * 1.08883], axis=-1)

    # XYZ -> 线性 RGB
    mat_inv = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ])
    rgb = xyz @ mat_inv.T

    # 线性 RGB -> sRGB
    rgb = np.clip(rgb, 0, None)
    mask = rgb > 0.0031308
    rgb[mask] = 1.055 * (rgb[mask] ** (1.0 / 2.4)) - 0.055
    rgb[~mask] = 12.92 * rgb[~mask]

    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return rgb


def align_brightness(test_img: np.ndarray, gt_img: np.ndarray) -> np.ndarray:
    """将测试图像的亮度对齐到 GT 图像

    在 LAB 空间中，将测试图像 L 通道的均值和标准差
    调整为与 GT 图像一致，保持色度通道不变。

    Args:
        test_img: uint8 RGB 测试图像
        gt_img: uint8 RGB GT 图像

    Returns:
        uint8 RGB 亮度对齐后的测试图像
    """
    test_lab = rgb_to_lab(test_img)
    gt_lab = rgb_to_lab(gt_img)

    test_l = test_lab[:, :, 0]
    gt_l = gt_lab[:, :, 0]

    # 计算均值和标准差
    test_mean, test_std = test_l.mean(), test_l.std()
    gt_mean, gt_std = gt_l.mean(), gt_l.std()

    # 避免除零
    if test_std < 1e-6:
        test_std = 1e-6

    # 线性变换：将测试图像 L 通道对齐到 GT
    aligned_l = (test_l - test_mean) * (gt_std / test_std) + gt_mean
    aligned_l = np.clip(aligned_l, 0, 100)

    test_lab[:, :, 0] = aligned_l
    return lab_to_rgb(test_lab)
