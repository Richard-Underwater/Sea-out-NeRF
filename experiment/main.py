"""颜色网格工具入口"""

colors = [
        "#e3ffff",  # 橙红
        "#82aaa9",  # 绿色
        "#739894",  # 蓝色
        "#577270",  # 粉色
        "#3d5757",  # 黄色
        "#21262a",  # 青色
        "#5c5345",  # 深橙
        "#cdd87a",  # 紫色
        "#7aae76",  # 纯绿
        "#72bbeb",
        "#2f3b46",
        "#59547a",
        "#434743",
        "#6e7958",
        "#548fa0",
        "#63728c",
        "#89a28f",
        "#7a907e"
    ]
colors_uw = [
        "#6b96c0",
        "#47728c",
        "#42677e",
        "#355770",
        "#2f4a61",
        "#223649",
        "#3d4d5b",
        "#628072",
        "#396f72",
        "#3d71a0",
        "#2b4058",
        "#3b496c",
        "#364453",
        "#3c585f",
        "#395f82",
        "#3b5478",
        "#4a677a",
        "#45606d"



    ]
colors_seathru= [
        "#d1d7df",
        "#71919a",
        "#5d7c85",
        "#475c60",
        "#354349",
        "#1f2327",
        "#4f4b43",
        "#8f9361",
        "#5a6f55",
        "#5b90af",
        "#2b353c",
        "#444c57",
        "#303131",
        "#484b4a",
        "#426374",
        "#4a5760",
        "#657170",
        "#5f6f70"
    ]

colors_gt = [
        "#fefcfa",
        "#c2eefd",
        "#aacfd5",
        "#71949b",
        "#526666",
        "#1f2324",
        "#8b5c49",
        "#fefe97",
        "#a2e99d",
        "#98f0fe",
        "#2f3f53",
        "#986aae",
        "#564740",
        "#9e8f63",
        "#6ab4d8",
        "#8882b8",
        "#d1d6be",
        "#b2bfa5"

    ]

import numpy as np


def hex_to_rgb(hex_str):
    """将 #RRGGBB 格式字符串转换为 [0, 1] 的 RGB 数组"""
    hex_str = hex_str.lstrip('#')
    return np.array([int(hex_str[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0


def rgb_to_lab(rgb):
    """sRGB 到 CIELAB 的转换 (简化版，无需额外库)"""

    # 1. 反伽马变换
    def gamma_inverse(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    rgb_lin = np.array([gamma_inverse(c) for c in rgb])

    # 2. XYZ 转换 (D50 参考白)
    matrix = np.array([
        [0.4360747, 0.3850649, 0.1430804],
        [0.2225045, 0.7168786, 0.0606169],
        [0.0139291, 0.0971035, 0.7141733]
    ])
    xyz = dot_product = np.dot(matrix, rgb_lin)

    # 3. XYZ 到 Lab
    def f(t):
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    # D50 白点坐标
    ref_white = np.array([0.9642, 1.0000, 0.8249])
    xyz_f = np.array([f(c) for c in (xyz / ref_white)])

    L = 116 * xyz_f[1] - 16
    a = 500 * (xyz_f[0] - xyz_f[1])
    b = 200 * (xyz_f[1] - xyz_f[2])
    return np.array([L, a, b])


def calculate_delta_e(lab1, lab2):
    """计算 CIE 1976 欧式色差 (用于演示，精确论文建议用 CIE 2000)"""
    return np.sqrt(np.sum((lab1 - lab2) ** 2))


def evaluate_test_sets(gt_list, test_sets):
    """
    gt_list: GT 的 RGB 字符串列表
    test_sets: 字典 {'Method1': [str...], 'Method2': [str...]}
    返回: (results, errors_dict)
    """
    gt_lab = [rgb_to_lab(hex_to_rgb(c)) for c in gt_list]

    results = {}
    errors_dict = {}

    for name, colors in test_sets.items():
        test_lab = [rgb_to_lab(hex_to_rgb(c)) for c in colors]
        errors = [calculate_delta_e(gt_lab[i], test_lab[i]) for i in range(len(gt_lab))]

        errors_dict[name] = errors
        results[name] = {
            "Mean ΔE": np.mean(errors),
            "Median ΔE": np.median(errors),
            "Max ΔE": np.max(errors),
            "RMSE": np.sqrt(np.mean(np.array(errors) ** 2))
        }

    return results, errors_dict


def plot_delta_e_trend(errors_dict, output_path="output/delta_e_trend.png"):
    """绘制各数据集色差趋势折线图"""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']

    for idx, (name, errors) in enumerate(errors_dict.items()):
        x = range(1, len(errors) + 1)
        plt.plot(
            x, errors,
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            linewidth=2,
            markersize=6,
            label=name
        )

    plt.xlabel('Color Patch Index', fontsize=12)
    plt.ylabel('ΔE (Color Difference)', fontsize=12)
    plt.title('Color Difference (ΔE) per Patch vs Ground Truth', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, len(next(iter(errors_dict.values()))) + 1))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"折线图已保存: {output_path}")


# --- 模拟数据输入 ---
# 请将这里替换为您提取出来的 4 个 lis

all_tests = {
    "Ours": colors,
    "Seathru": colors_seathru,
    "UW": colors_uw
}

# 执行评估
metrics, errors_dict = evaluate_test_sets(colors_gt, all_tests)

# --- 打印结果表格 ---
print(f"{'Method':<25} | {'Mean ΔE ↓':<10} | {'Median ΔE':<10} | {'Max ΔE':<10}")
print("-" * 65)
for method, vals in metrics.items():
    print(f"{method:<25} | {vals['Mean ΔE']:<10.2f} | {vals['Median ΔE']:<10.2f} | {vals['Max ΔE']:<10.2f}")

# --- 绘制折线图 ---
plot_delta_e_trend(errors_dict)