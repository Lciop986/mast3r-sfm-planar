#!/user/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 2026/6/27 09:59
# @Author: 23263
import os
import sys
import json
import numpy as np
from argparse import ArgumentParser


def parse_cloudcompare_txt_adaptive(txt_path):
    print(f">> 正在解析 CloudCompare 导出文件: {txt_path}")

    # 1. 首先读取第一行，动态识别“距离列”所在的索引
    target_col_idx = None
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        header = f.readline().strip()

    if not header.startswith('//'):
        sys.exit("❌ 错误：文件第一行不是合法的 CloudCompare 注释表头！")

    # 切分表头单词
    header_words = header.replace('//', '').split()
    target_name = "C2C_absolute_distances"

    if target_name in header_words:
        target_col_idx = header_words.index(target_name)
        print(f"🔍 自动检测成功：'{target_name}' 位于第 {target_col_idx + 1} 列。")
    else:
        sys.exit(f"❌ 错误：在表头中未找到名为 '{target_name}' 的列，请检查 CloudCompare 是否成功计算了测距。")

    # 2. 动用 numpy 底层 C 算子，精准提取该列
    print(">> 底层算子高速并行读取中...")
    try:
        distances = np.loadtxt(txt_path, skiprows=2, usecols=(target_col_idx,))
    except Exception as e:
        print(f"⚠️ 高速读取失败 ({e})，正在切换至大文件流式兼容解析...")
        distances = []
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            for idx, line in enumerate(f):
                if idx < 2 or line.startswith('//') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) > target_col_idx:
                    distances.append(float(parts[target_col_idx]))
        distances = np.array(distances)

    total_points = len(distances)
    print(f"🎉 成功载入数据！有效计算点数: {total_points} 个。")

    if total_points == 0:
        sys.exit("❌ 错误：未能提取到任何有效的距离数据。")

    # 3. 统计核心量化指标
    mean_err = np.mean(distances)
    median_err = np.median(distances)
    rmse = np.sqrt(np.mean(distances ** 2))

    percentile_90 = np.percentile(distances, 90)
    percentile_95 = np.percentile(distances, 95)

    inliers_20cm = np.sum(distances < 0.20) / total_points
    inliers_10cm = np.sum(distances < 0.10) / total_points

    # 统一转换为 cm 和 %
    metrics = {
        "Total_Points_Evaluated": int(total_points),
        "Mean_Distance_Error (cm)": float(mean_err * 100),
        "Median_Distance_Error (cm)": float(median_err * 100),
        "RMSE (cm)": float(rmse * 100),
        "90%_Percentile_Error (cm)": float(percentile_90 * 100),
        "95%_Percentile_Error (cm)": float(percentile_95 * 100),
        "★_20cm_Inlier_Ratio (%)": float(inliers_20cm * 100),
        "★_10cm_Inlier_Ratio (Bonus %)": float(inliers_10cm * 100)
    }

    return metrics


def main():
    parser = ArgumentParser(description="CloudCompare 智能自适应指标提取工具")
    parser.add_argument('--input', '-i', required=True, help="CloudCompare 导出的 .txt 文件路径")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"❌ 错误：找不到文件 {args.input}")

    metrics = parse_cloudcompare_txt_adaptive(args.input)

    print(f"  ● 总参评点数 (Total Points):    {metrics['Total_Points_Evaluated']} 个")
    print(f"  ● 平均距离误差 (Mean Error):     {metrics['Mean_Distance_Error (cm)']:.3f} cm")
    print(f"  ● 中位数距离误差 (Median Error):  {metrics['Median_Distance_Error (cm)']:.3f} cm")
    print(f"  ● 均方根误差 (RMSE):             {metrics['RMSE (cm)']:.3f} cm")
    print(f"  ● 90% 分位空间误差:             {metrics['90%_Percentile_Error (cm)']:.3f} cm")
    print(f"  ● 95% 分位空间误差:             {metrics['95%_Percentile_Error (cm)']:.3f} cm")
    print(f"  ---------------------------------------------------------")
    print(f"  ★ 20 cm 内点比例 (基础目标线):   {metrics['★_20cm_Inlier_Ratio (%)']:.2f} %")
    print(f"  ★ 10 cm 内点比例 (绝对加分项!):  {metrics['★_10cm_Inlier_Ratio (Bonus %)']:.2f} %")
    print("█" * 68 + "\n")

    # 在当前目录下导出一份 JSON 成果物文件
    output_json = os.path.splitext(args.input)[0] + "_gdc_metrics.json"
    with open(output_json, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f">> [成功] 自适应量化数据表已保存至: {output_json}\n")


if __name__ == '__main__':
    main()