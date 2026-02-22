"""
图像质量评估脚本
计算测试图像与GT之间的 PSNR, SSIM, LPIPS 指标

目录结构:
input/
├── {场景1}/
│   ├── test/
│   │   ├── {消融实验1}/
│   │   ├── {消融实验2}/
│   │   └── ...
│   └── GT/
├── {场景2}/
│   ├── test/
│   │   ├── {消融实验1}/
│   │   ├── {消融实验2}/
│   │   └── ...
│   └── GT/
├── metrics_cross_scenes.csv      # 跨场景汇总
└── metrics_cross_scenes.txt      # 跨场景汇总文本
"""

import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def load_image(path):
    """加载图像并转换为numpy数组"""
    img = Image.open(path).convert('RGB')
    return np.array(img)


def load_image_tensor(path, device):
    """加载图像并转换为LPIPS所需的tensor格式 [-1, 1]"""
    img = Image.open(path).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0
    img = img * 2 - 1  # 归一化到 [-1, 1]
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return img


def calculate_psnr(img1, img2):
    """计算PSNR"""
    return psnr(img1, img2, data_range=255)


def calculate_ssim(img1, img2):
    """计算SSIM"""
    return ssim(img1, img2, channel_axis=2, data_range=255)


def calculate_lpips(img1_tensor, img2_tensor, lpips_fn):
    """计算LPIPS"""
    with torch.no_grad():
        lpips_value = lpips_fn(img1_tensor, img2_tensor)
    return lpips_value.item()


def get_image_pairs(test_dir, gt_dir):
    """获取测试图像和GT图像的配对列表"""
    test_files = set(os.listdir(test_dir))
    gt_files = set(os.listdir(gt_dir))

    # 支持的图像格式
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}

    # 过滤有效图像文件
    test_images = {f for f in test_files if os.path.splitext(f)[1].lower() in valid_extensions}
    gt_images = {f for f in gt_files if os.path.splitext(f)[1].lower() in valid_extensions}

    # 找到匹配的图像对（文件名相同）
    common_files = test_images & gt_images

    if not common_files:
        # 如果文件名不完全相同，尝试按名称排序配对
        test_sorted = sorted(test_images)
        gt_sorted = sorted(gt_images)
        if len(test_sorted) == len(gt_sorted):
            print("警告: 文件名不匹配，按排序顺序配对")
            return list(zip(test_sorted, gt_sorted))
        else:
            raise ValueError(f"测试图像数量({len(test_sorted)})与GT图像数量({len(gt_sorted)})不匹配")

    return [(f, f) for f in sorted(common_files)]


def get_subdirs(directory):
    """获取目录下的所有子目录"""
    if not os.path.exists(directory):
        return []

    subdirs = []
    for name in os.listdir(directory):
        full_path = os.path.join(directory, name)
        if os.path.isdir(full_path):
            subdirs.append(name)

    return sorted(subdirs)


def get_scenes(base_dir):
    """获取所有场景目录（包含test和GT子目录的目录）"""
    scenes = []
    for name in get_subdirs(base_dir):
        scene_dir = os.path.join(base_dir, name)
        test_dir = os.path.join(scene_dir, 'test')
        gt_dir = os.path.join(scene_dir, 'GT')
        if os.path.exists(test_dir) and os.path.exists(gt_dir):
            scenes.append(name)
    return sorted(scenes)


def evaluate_single(test_dir, gt_dir, lpips_fn, device):
    """评估单个消融实验"""
    # 获取图像对
    image_pairs = get_image_pairs(test_dir, gt_dir)

    if not image_pairs:
        return None, []

    # 存储结果
    results = []
    psnr_values = []
    ssim_values = []
    lpips_values = []

    # 计算每张图像的指标
    for test_name, gt_name in tqdm(image_pairs, desc="计算指标", leave=False):
        test_path = os.path.join(test_dir, test_name)
        gt_path = os.path.join(gt_dir, gt_name)

        # 加载图像
        test_img = load_image(test_path)
        gt_img = load_image(gt_path)

        # 检查尺寸是否一致
        if test_img.shape != gt_img.shape:
            print(f"警告: 图像尺寸不匹配 {test_name}: {test_img.shape} vs {gt_img.shape}")
            # 调整测试图像尺寸以匹配GT
            test_pil = Image.open(test_path).convert('RGB')
            test_pil = test_pil.resize((gt_img.shape[1], gt_img.shape[0]), Image.LANCZOS)
            test_img = np.array(test_pil)

        # 计算PSNR和SSIM
        psnr_val = calculate_psnr(gt_img, test_img)
        ssim_val = calculate_ssim(gt_img, test_img)

        # 计算LPIPS
        test_tensor = load_image_tensor(test_path, device)
        gt_tensor = load_image_tensor(gt_path, device)

        # 如果尺寸不匹配，调整tensor
        if test_tensor.shape != gt_tensor.shape:
            test_tensor = torch.nn.functional.interpolate(
                test_tensor, size=(gt_tensor.shape[2], gt_tensor.shape[3]),
                mode='bilinear', align_corners=False
            )

        lpips_val = calculate_lpips(test_tensor, gt_tensor, lpips_fn)

        # 记录结果
        results.append({
            'image': test_name,
            'PSNR': psnr_val,
            'SSIM': ssim_val,
            'LPIPS': lpips_val
        })

        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
        lpips_values.append(lpips_val)

    # 计算平均值
    avg_metrics = {
        'PSNR': np.mean(psnr_values),
        'SSIM': np.mean(ssim_values),
        'LPIPS': np.mean(lpips_values),
        'count': len(image_pairs)
    }

    # 添加平均值行
    results.append({
        'image': 'Average',
        'PSNR': avg_metrics['PSNR'],
        'SSIM': avg_metrics['SSIM'],
        'LPIPS': avg_metrics['LPIPS']
    })

    return avg_metrics, results


def evaluate_all_scenes(base_dir='./input'):
    """评估所有场景的所有消融实验，并计算跨场景平均指标"""

    # 获取所有场景
    scenes = get_scenes(base_dir)
    if not scenes:
        raise ValueError(f"未找到有效场景目录（需包含test和GT子目录）: {base_dir}")

    print(f"找到 {len(scenes)} 个场景: {scenes}")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 初始化LPIPS模型
    print("加载LPIPS模型...")
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    lpips_fn.eval()

    # 收集所有消融实验名称
    all_ablations = set()
    for scene in scenes:
        test_dir = os.path.join(base_dir, scene, 'test')
        ablations = get_subdirs(test_dir)
        all_ablations.update(ablations)

    all_ablations = sorted(all_ablations)
    print(f"找到 {len(all_ablations)} 个消融实验: {all_ablations}")

    # 存储结果: {ablation: {scene: metrics}}
    all_results = {abl: {} for abl in all_ablations}

    # 遍历每个场景
    for scene in scenes:
        print(f"\n{'='*60}")
        print(f"场景: {scene}")
        print('='*60)

        scene_dir = os.path.join(base_dir, scene)
        test_base_dir = os.path.join(scene_dir, 'test')
        gt_dir = os.path.join(scene_dir, 'GT')

        scene_summary = []

        # 遍历该场景下的消融实验
        ablations = get_subdirs(test_base_dir)
        for abl_name in ablations:
            print(f"\n  消融实验: {abl_name}")

            test_dir = os.path.join(test_base_dir, abl_name)
            avg_metrics, results = evaluate_single(test_dir, gt_dir, lpips_fn, device)

            if avg_metrics is None:
                print(f"    警告: 未找到匹配的图像对，跳过")
                continue

            # 保存单个消融实验的详细CSV
            df = pd.DataFrame(results)
            csv_path = os.path.join(scene_dir, f'metrics_{abl_name}.csv')
            df.to_csv(csv_path, index=False, float_format='%.6f')

            # 记录结果
            all_results[abl_name][scene] = avg_metrics

            scene_summary.append({
                'ablation': abl_name,
                'PSNR': avg_metrics['PSNR'],
                'SSIM': avg_metrics['SSIM'],
                'LPIPS': avg_metrics['LPIPS'],
                'image_count': avg_metrics['count']
            })

            print(f"    PSNR: {avg_metrics['PSNR']:.4f} | SSIM: {avg_metrics['SSIM']:.4f} | LPIPS: {avg_metrics['LPIPS']:.4f}")

        # 保存场景级汇总
        if scene_summary:
            scene_df = pd.DataFrame(scene_summary)
            scene_csv_path = os.path.join(scene_dir, 'metrics_summary.csv')
            scene_df.to_csv(scene_csv_path, index=False, float_format='%.6f')

    # 计算跨场景平均指标
    print(f"\n{'='*60}")
    print("计算跨场景平均指标...")
    print('='*60)

    cross_scene_results = []

    for abl_name in all_ablations:
        scene_metrics = all_results[abl_name]
        if not scene_metrics:
            continue

        # 计算该消融实验在所有场景的平均值
        psnr_vals = [m['PSNR'] for m in scene_metrics.values()]
        ssim_vals = [m['SSIM'] for m in scene_metrics.values()]
        lpips_vals = [m['LPIPS'] for m in scene_metrics.values()]

        avg_result = {
            'ablation': abl_name,
            'PSNR': np.mean(psnr_vals),
            'SSIM': np.mean(ssim_vals),
            'LPIPS': np.mean(lpips_vals),
            'scene_count': len(scene_metrics)
        }

        # 添加每个场景的指标
        for scene in scenes:
            if scene in scene_metrics:
                avg_result[f'{scene}_PSNR'] = scene_metrics[scene]['PSNR']
                avg_result[f'{scene}_SSIM'] = scene_metrics[scene]['SSIM']
                avg_result[f'{scene}_LPIPS'] = scene_metrics[scene]['LPIPS']
            else:
                avg_result[f'{scene}_PSNR'] = np.nan
                avg_result[f'{scene}_SSIM'] = np.nan
                avg_result[f'{scene}_LPIPS'] = np.nan

        cross_scene_results.append(avg_result)

        print(f"\n[{abl_name}]")
        print(f"  平均 PSNR: {avg_result['PSNR']:.4f} dB")
        print(f"  平均 SSIM: {avg_result['SSIM']:.4f}")
        print(f"  平均 LPIPS: {avg_result['LPIPS']:.4f}")

    # 保存跨场景汇总到input根目录
    if cross_scene_results:
        # CSV格式
        cross_df = pd.DataFrame(cross_scene_results)
        # 重新排列列顺序
        cols = ['ablation', 'PSNR', 'SSIM', 'LPIPS', 'scene_count']
        for scene in scenes:
            cols.extend([f'{scene}_PSNR', f'{scene}_SSIM', f'{scene}_LPIPS'])
        cross_df = cross_df[[c for c in cols if c in cross_df.columns]]

        cross_csv_path = os.path.join(base_dir, 'metrics_cross_scenes.csv')
        cross_df.to_csv(cross_csv_path, index=False, float_format='%.6f')

        # 文本格式
        summary_text = f"""
{'='*70}
                    跨场景评估结果汇总
{'='*70}
场景列表: {', '.join(scenes)}
消融实验数量: {len(cross_scene_results)}

"""
        # 表格头
        summary_text += f"{'消融实验':<20} {'PSNR':>10} {'SSIM':>10} {'LPIPS':>10}\n"
        summary_text += "-" * 55 + "\n"

        for res in cross_scene_results:
            summary_text += f"{res['ablation']:<20} {res['PSNR']:>10.4f} {res['SSIM']:>10.4f} {res['LPIPS']:>10.4f}\n"

        summary_text += "\n" + "=" * 70 + "\n"
        summary_text += "\n各场景详细指标:\n"
        summary_text += "=" * 70 + "\n"

        for res in cross_scene_results:
            summary_text += f"\n[{res['ablation']}]\n"
            for scene in scenes:
                psnr_key = f'{scene}_PSNR'
                ssim_key = f'{scene}_SSIM'
                lpips_key = f'{scene}_LPIPS'
                if psnr_key in res and not np.isnan(res[psnr_key]):
                    summary_text += f"  {scene}: PSNR={res[psnr_key]:.4f}, SSIM={res[ssim_key]:.4f}, LPIPS={res[lpips_key]:.4f}\n"

        summary_path = os.path.join(base_dir, 'metrics_cross_scenes.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)

        print(f"\n跨场景汇总结果已保存到:")
        print(f"  - {cross_csv_path}")
        print(f"  - {summary_path}")

    return cross_scene_results


def evaluate_scene(scene_name, base_dir='./input', ablation_name=None):
    """评估单个场景的消融实验"""
    scene_dir = os.path.join(base_dir, scene_name)
    test_base_dir = os.path.join(scene_dir, 'test')
    gt_dir = os.path.join(scene_dir, 'GT')

    # 检查目录是否存在
    if not os.path.exists(test_base_dir):
        raise FileNotFoundError(f"测试目录不存在: {test_base_dir}")
    if not os.path.exists(gt_dir):
        raise FileNotFoundError(f"GT目录不存在: {gt_dir}")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 初始化LPIPS模型
    print("加载LPIPS模型...")
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    lpips_fn.eval()

    # 获取要评估的消融实验
    if ablation_name:
        ablation_dirs = [ablation_name]
    else:
        ablation_dirs = get_subdirs(test_base_dir)

    if not ablation_dirs:
        raise ValueError(f"未找到消融实验目录: {test_base_dir}")

    print(f"找到 {len(ablation_dirs)} 个消融实验: {ablation_dirs}")

    summary_results = []

    for abl_name in ablation_dirs:
        print(f"\n{'='*50}")
        print(f"评估消融实验: {abl_name}")
        print('='*50)

        test_dir = os.path.join(test_base_dir, abl_name)

        if not os.path.exists(test_dir):
            print(f"警告: 目录不存在，跳过: {test_dir}")
            continue

        avg_metrics, results = evaluate_single(test_dir, gt_dir, lpips_fn, device)

        if avg_metrics is None:
            print(f"警告: 未找到匹配的图像对，跳过: {abl_name}")
            continue

        # 保存详细CSV
        df = pd.DataFrame(results)
        csv_path = os.path.join(scene_dir, f'metrics_{abl_name}.csv')
        df.to_csv(csv_path, index=False, float_format='%.6f')
        print(f"详细结果已保存到: {csv_path}")

        summary_results.append({
            'ablation': abl_name,
            'PSNR': avg_metrics['PSNR'],
            'SSIM': avg_metrics['SSIM'],
            'LPIPS': avg_metrics['LPIPS'],
            'image_count': avg_metrics['count']
        })

        print(f"  PSNR:  {avg_metrics['PSNR']:.4f} dB")
        print(f"  SSIM:  {avg_metrics['SSIM']:.4f}")
        print(f"  LPIPS: {avg_metrics['LPIPS']:.4f}")

    # 保存汇总
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_csv_path = os.path.join(scene_dir, 'metrics_summary.csv')
        summary_df.to_csv(summary_csv_path, index=False, float_format='%.6f')
        print(f"\n汇总结果已保存到: {summary_csv_path}")

    return summary_results


def main():
    parser = argparse.ArgumentParser(description='计算图像质量评估指标 (PSNR, SSIM, LPIPS)')
    parser.add_argument('--scene', '-s', type=str, default=None,
                        help='场景名称 (不指定则评估所有场景)')
    parser.add_argument('--ablation', '-a', type=str, default=None,
                        help='消融实验名称 (不指定则评估所有)')
    parser.add_argument('--base_dir', '-b', type=str, default='./input',
                        help='基础目录路径 (默认: ./input)')

    args = parser.parse_args()

    if args.scene:
        # 评估单个场景
        evaluate_scene(args.scene, args.base_dir, args.ablation)
    else:
        # 评估所有场景并计算跨场景平均
        evaluate_all_scenes(args.base_dir)


if __name__ == '__main__':
    main()
