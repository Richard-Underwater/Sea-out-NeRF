from __future__ import annotations

"""
去水印图像质量评估模块
计算 remove_water/{方法名} 与 remove_water/GT 之间的图像差异度

流程:
1. 将待测试图像的亮度对齐到 GT 图像
2. 计算 PSNR, SSIM, LPIPS

目录结构:
remove_water/
├── GT/              # 真值图像
├── {方法1}/         # 方法1的去水印结果
├── {方法2}/         # 方法2的去水印结果
└── metrics/         # 输出的评估结果
"""

import os
import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from src.brightness_align import align_brightness


@dataclass
class ImageMetrics:
    """单张图像的评估指标"""
    image_name: str
    psnr: float
    ssim: float
    lpips: float


@dataclass
class MethodMetrics:
    """单个方法的汇总指标"""
    method: str
    avg_psnr: float
    avg_ssim: float
    avg_lpips: float
    image_count: int


VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}


def _list_images(directory: str) -> list[str]:
    """列出目录中的所有有效图像文件名"""
    return sorted(
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
    )


def _match_image_pairs(
    test_dir: str, gt_dir: str
) -> list[tuple[str, str]]:
    """匹配测试目录和 GT 目录中的图像对"""
    test_images = set(_list_images(test_dir))
    gt_images = set(_list_images(gt_dir))

    common = test_images & gt_images
    if common:
        return [(f, f) for f in sorted(common)]

    # 文件名不匹配时按排序配对
    test_sorted = sorted(test_images)
    gt_sorted = sorted(gt_images)
    if len(test_sorted) == len(gt_sorted):
        print("警告: 文件名不匹配，按排序顺序配对")
        return list(zip(test_sorted, gt_sorted))

    raise ValueError(
        f"图像数量不匹配: 测试={len(test_sorted)}, GT={len(gt_sorted)}"
    )


def _load_image(path: str) -> np.ndarray:
    """加载图像为 uint8 RGB numpy 数组"""
    return np.array(Image.open(path).convert('RGB'))


def _to_lpips_tensor(
    img: np.ndarray, device: torch.device
) -> torch.Tensor:
    """将 uint8 RGB 图像转为 LPIPS 所需的 tensor [-1, 1]"""
    t = img.astype(np.float32) / 255.0 * 2 - 1
    return torch.from_numpy(t).permute(2, 0, 1).unsqueeze(0).to(device)


def _resize_to_match(
    test_img: np.ndarray, gt_shape: tuple[int, ...]
) -> np.ndarray:
    """将测试图像调整到与 GT 相同的尺寸"""
    h, w = gt_shape[:2]
    pil = Image.fromarray(test_img).resize((w, h), Image.LANCZOS)
    return np.array(pil)


def evaluate_method(
    method_dir: str,
    gt_dir: str,
    lpips_fn: lpips.LPIPS,
    device: torch.device,
    use_brightness_align: bool = True,
) -> tuple[MethodMetrics | None, list[ImageMetrics]]:
    """评估单个方法与 GT 的差异度"""
    pairs = _match_image_pairs(method_dir, gt_dir)
    if not pairs:
        return None, []

    results: list[ImageMetrics] = []

    for test_name, gt_name in tqdm(pairs, desc="计算指标", leave=False):
        test_path = os.path.join(method_dir, test_name)
        gt_path = os.path.join(gt_dir, gt_name)

        test_img = _load_image(test_path)
        gt_img = _load_image(gt_path)

        # 尺寸对齐
        if test_img.shape != gt_img.shape:
            print(f"警告: 尺寸不匹配 {test_name}, 自动调整")
            test_img = _resize_to_match(test_img, gt_img.shape)

        # 亮度对齐
        if use_brightness_align:
            test_img = align_brightness(test_img, gt_img)

        # PSNR & SSIM
        psnr_val = psnr(gt_img, test_img, data_range=255)
        ssim_val = ssim(gt_img, test_img, channel_axis=2, data_range=255)

        # LPIPS
        test_tensor = _to_lpips_tensor(test_img, device)
        gt_tensor = _to_lpips_tensor(gt_img, device)
        with torch.no_grad():
            lpips_val = lpips_fn(test_tensor, gt_tensor).item()

        results.append(ImageMetrics(test_name, psnr_val, ssim_val, lpips_val))

    avg = MethodMetrics(
        method=os.path.basename(method_dir),
        avg_psnr=np.mean([r.psnr for r in results]),
        avg_ssim=np.mean([r.ssim for r in results]),
        avg_lpips=np.mean([r.lpips for r in results]),
        image_count=len(results),
    )
    return avg, results


def evaluate_all(
    base_dir: str = '.\\remove_water',
    use_brightness_align: bool = True,
) -> list[MethodMetrics]:
    """评估 remove_water 目录下所有方法"""
    gt_dir = os.path.join(base_dir, 'GT')
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"GT 目录不存在: {gt_dir}")

    # 找出所有方法目录（排除 GT 和 metrics）
    methods = sorted(
        name for name in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, name))
        and name not in ('GT', 'metrics')
    )
    if not methods:
        raise ValueError(f"未找到任何方法目录: {base_dir}")

    align_label = "开启" if use_brightness_align else "关闭"
    print(f"亮度对齐: {align_label}")
    print(f"GT 目录: {gt_dir}")
    print(f"待评估方法: {methods}")

    # 初始化设备和 LPIPS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print("加载 LPIPS 模型...")
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    lpips_fn.eval()

    # 输出目录
    metrics_dir = os.path.join(base_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    summaries: list[MethodMetrics] = []

    for method in methods:
        print(f"\n{'='*50}")
        print(f"方法: {method}")
        print('='*50)

        method_dir = os.path.join(base_dir, method)
        avg, details = evaluate_method(
            method_dir, gt_dir, lpips_fn, device, use_brightness_align
        )

        if avg is None:
            print("  警告: 未找到匹配的图像对，跳过")
            continue

        # 保存单方法详细结果
        detail_rows = [
            {'image': r.image_name, 'PSNR': r.psnr,
             'SSIM': r.ssim, 'LPIPS': r.lpips}
            for r in details
        ]
        detail_rows.append({
            'image': 'Average',
            'PSNR': avg.avg_psnr, 'SSIM': avg.avg_ssim,
            'LPIPS': avg.avg_lpips,
        })
        df = pd.DataFrame(detail_rows)
        csv_path = os.path.join(metrics_dir, f'{method}.csv')
        df.to_csv(csv_path, index=False, float_format='%.6f')

        summaries.append(avg)
        print(f"  PSNR:  {avg.avg_psnr:.4f} dB")
        print(f"  SSIM:  {avg.avg_ssim:.4f}")
        print(f"  LPIPS: {avg.avg_lpips:.4f}")

    _save_summary(summaries, metrics_dir, use_brightness_align)
    return summaries


def generate_aligned_images(
    base_dir: str = '.\\remove_water',
) -> None:
    """生成亮度对齐后的测试图像，保存到 {base_dir}/{方法名}_aligned/"""
    gt_dir = os.path.join(base_dir, 'GT')
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"GT 目录不存在: {gt_dir}")

    methods = sorted(
        name for name in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, name))
        and name not in ('GT', 'metrics')
        and not name.endswith('_aligned')
    )
    if not methods:
        raise ValueError(f"未找到任何方法目录: {base_dir}")

    print(f"待处理方法: {methods}")

    for method in methods:
        method_dir = os.path.join(base_dir, method)
        output_dir = os.path.join(base_dir, f'{method}_aligned')
        os.makedirs(output_dir, exist_ok=True)

        pairs = _match_image_pairs(method_dir, gt_dir)
        print(f"\n方法: {method} ({len(pairs)} 张图像)")

        for test_name, gt_name in tqdm(pairs, desc=f"对齐 {method}"):
            test_img = _load_image(os.path.join(method_dir, test_name))
            gt_img = _load_image(os.path.join(gt_dir, gt_name))

            if test_img.shape != gt_img.shape:
                test_img = _resize_to_match(test_img, gt_img.shape)

            aligned = align_brightness(test_img, gt_img)
            Image.fromarray(aligned).save(os.path.join(output_dir, test_name))

        print(f"  已保存到: {output_dir}")


def _save_summary(
    summaries: list[MethodMetrics], output_dir: str,
    use_brightness_align: bool = True,
) -> None:
    """保存汇总结果（CSV + TXT）"""
    if not summaries:
        return

    rows = [
        {'method': s.method, 'PSNR': s.avg_psnr,
         'SSIM': s.avg_ssim, 'LPIPS': s.avg_lpips,
         'image_count': s.image_count}
        for s in summaries
    ]
    df = pd.DataFrame(rows)

    csv_path = os.path.join(output_dir, 'summary.csv')
    df.to_csv(csv_path, index=False, float_format='%.6f')

    # 文本格式
    align_text = "亮度对齐后" if use_brightness_align else "未亮度对齐"
    lines = [
        '=' * 60,
        f'       去水印图像质量评估汇总 ({align_text})',
        '=' * 60,
        '',
        f"{'方法':<20} {'PSNR':>10} {'SSIM':>10} {'LPIPS':>10}",
        '-' * 55,
    ]
    for s in summaries:
        lines.append(
            f"{s.method:<20} {s.avg_psnr:>10.4f} "
            f"{s.avg_ssim:>10.4f} {s.avg_lpips:>10.4f}"
        )
    lines.append('=' * 60)

    txt_path = os.path.join(output_dir, 'summary.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\n汇总结果已保存:")
    print(f"  - {csv_path}")
    print(f"  - {txt_path}")


def main():
    parser = argparse.ArgumentParser(
        description='去水印图像质量评估 (PSNR/SSIM/LPIPS, 含亮度对齐)'
    )
    parser.add_argument(
        '--base_dir', '-b', type=str, default='./remove_water',
        help='基础目录路径 (默认: ./remove_water)'
    )
    parser.add_argument(
        '--method', '-m', type=str, default=None,
        help='指定单个方法名称 (不指定则评估所有方法)'
    )
    parser.add_argument(
        '--no-align', action='store_true',
        help='关闭亮度对齐 (默认开启)'
    )
    parser.add_argument(
        '--gen-aligned', action='store_true',
        help='仅生成亮度对齐后的图像，不计算指标'
    )
    args = parser.parse_args()
    use_align = not args.no_align

    if args.gen_aligned:
        generate_aligned_images(args.base_dir)
    elif args.method:
        # 评估单个方法
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        lpips_fn = lpips.LPIPS(net='alex').to(device)
        lpips_fn.eval()

        method_dir = os.path.join(args.base_dir, args.method)
        gt_dir = os.path.join(args.base_dir, 'GT')
        avg, details = evaluate_method(
            method_dir, gt_dir, lpips_fn, device, use_align
        )

        if avg:
            print(f"\nPSNR:  {avg.avg_psnr:.4f} dB")
            print(f"SSIM:  {avg.avg_ssim:.4f}")
            print(f"LPIPS: {avg.avg_lpips:.4f}")
    else:
        evaluate_all(args.base_dir, use_align)


if __name__ == '__main__':
    main()
