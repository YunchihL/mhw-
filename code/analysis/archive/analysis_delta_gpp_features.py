#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 delta_gpp < 0 样本的特征
比较正向响应（delta_gpp < 0）和负向响应（delta_gpp > 0）样本的特征差异
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 路径配置
EVENT_ROWS_PATH = "analysis/results/step5q_event_rows.csv"
DATA_CSV_PATH = "data/data.csv"
OUTPUT_DIR = "analysis/results/delta_gpp_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("=" * 80)
    print("分析 delta_gpp < 0 样本的特征")
    print("=" * 80)

    # 1. 加载数据
    print("[1] 加载数据...")
    df_events = pd.read_csv(EVENT_ROWS_PATH)
    df_data = pd.read_csv(DATA_CSV_PATH)

    print(f"MHW事件月数量: {len(df_events)}")
    print(f"原始数据行数: {len(df_data)}")

    # 2. 合并数据
    print("[2] 合并数据...")
    # 选择需要的特征列
    feature_cols = [
        # 基础信息
        "grid_id", "year", "month", "lat_c", "lon_c",
        # 气候变量
        "pr_mm", "srad_mj_m2_day", "tavg_celsius", "tmmx_celsius",
        "tmmn_celsius", "vpd_pa", "ws_mps", "NDVI_avg", "mean_sst",
        # MHW特征
        "intensity_max_month", "duration_weighted_sum",
        "intensity_cumulative_weighted_sum", "intensity_density",
        # 其他
        "gpp_total", "gpp_mean_rate", "mangrove_area"
    ]

    # 过滤存在的列
    available_cols = [col for col in feature_cols if col in df_data.columns]
    print(f"可用的特征列: {len(available_cols)}/{len(feature_cols)}")

    # 合并
    df = pd.merge(
        df_events,
        df_data[available_cols],
        on=["grid_id", "year", "month"],
        how="left"
    )

    print(f"合并后数据行数: {len(df)}")

    # 3. 分割样本
    print("[3] 分割样本...")
    # 正向响应: delta_gpp < 0
    pos_samples = df[df["delta_gpp"] < 0].copy()
    # 负向响应: delta_gpp > 0
    neg_samples = df[df["delta_gpp"] > 0].copy()
    # 中性: delta_gpp == 0 (应该很少)
    neutral_samples = df[df["delta_gpp"] == 0].copy()

    print(f"正向响应样本 (ΔGPP < 0): {len(pos_samples)} ({len(pos_samples)/len(df)*100:.1f}%)")
    print(f"负向响应样本 (ΔGPP > 0): {len(neg_samples)} ({len(neg_samples)/len(df)*100:.1f}%)")
    print(f"中性样本 (ΔGPP = 0): {len(neutral_samples)} ({len(neutral_samples)/len(df)*100:.1f}%)")

    # 4. 基础统计
    print("\n[4] ΔGPP 统计:")
    print(f"ΔGPP 整体均值: {df['delta_gpp'].mean():.2e}")
    print(f"ΔGPP 整体中位数: {df['delta_gpp'].median():.2e}")
    print(f"ΔGPP 整体标准差: {df['delta_gpp'].std():.2e}")
    print(f"正向响应 ΔGPP 均值: {pos_samples['delta_gpp'].mean():.2e}")
    print(f"负向响应 ΔGPP 均值: {neg_samples['delta_gpp'].mean():.2e}")

    # 5. 特征比较
    print("\n[5] 特征比较 (正向 vs 负向):")

    # 选择要比较的特征
    compare_features = [
        # 气候变量
        "pr_mm", "srad_mj_m2_day", "tavg_celsius", "tmmx_celsius",
        "tmmn_celsius", "vpd_pa", "ws_mps", "NDVI_avg", "mean_sst",
        # MHW特征
        "intensity_max_month", "duration_weighted_sum",
        "intensity_cumulative_weighted_sum", "intensity_density",
        # GPP相关
        "gpp_total", "gpp_mean_rate"
    ]

    # 过滤存在的特征
    compare_features = [f for f in compare_features if f in df.columns]

    results = []
    for feature in compare_features:
        # 移除缺失值
        pos_vals = pos_samples[feature].dropna()
        neg_vals = neg_samples[feature].dropna()

        if len(pos_vals) > 10 and len(neg_vals) > 10:
            # 计算统计量
            pos_mean = pos_vals.mean()
            neg_mean = neg_vals.mean()
            pos_median = pos_vals.median()
            neg_median = neg_vals.median()

            # t检验
            t_stat, p_value = stats.ttest_ind(pos_vals, neg_vals, equal_var=False)

            # 效应量 (Cohen's d)
            pooled_std = np.sqrt((pos_vals.std()**2 + neg_vals.std()**2) / 2)
            cohens_d = (pos_mean - neg_mean) / pooled_std if pooled_std != 0 else 0

            results.append({
                "feature": feature,
                "pos_mean": pos_mean,
                "neg_mean": neg_mean,
                "pos_median": pos_median,
                "neg_median": neg_median,
                "mean_diff": pos_mean - neg_mean,
                "t_stat": t_stat,
                "p_value": p_value,
                "cohens_d": cohens_d,
                "pos_n": len(pos_vals),
                "neg_n": len(neg_vals)
            })

    # 创建结果DataFrame
    results_df = pd.DataFrame(results)

    # 按p值排序
    results_df = results_df.sort_values("p_value")

    # 6. 输出结果
    print("\n[6] 特征差异统计检验:")
    print("-" * 100)
    print(f"{'特征':<25} {'正向均值':<12} {'负向均值':<12} {'差异':<12} {'t统计量':<10} {'p值':<10} {'效应量':<8}")
    print("-" * 100)

    for _, row in results_df.iterrows():
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
        print(f"{row['feature']:<25} {row['pos_mean']:<12.3f} {row['neg_mean']:<12.3f} "
              f"{row['mean_diff']:<12.3f} {row['t_stat']:<10.3f} {row['p_value']:<10.3e} {row['cohens_d']:<8.3f}{sig}")

    print("-" * 100)

    # 7. 时空分布
    print("\n[7] 时空分布:")

    # 月份分布
    pos_month_counts = pos_samples["month"].value_counts().sort_index()
    neg_month_counts = neg_samples["month"].value_counts().sort_index()

    month_df = pd.DataFrame({
        "month": range(1, 13),
        "pos_count": [pos_month_counts.get(m, 0) for m in range(1, 13)],
        "neg_count": [neg_month_counts.get(m, 0) for m in range(1, 13)]
    })

    month_df["pos_frac"] = month_df["pos_count"] / month_df["pos_count"].sum()
    month_df["neg_frac"] = month_df["neg_count"] / month_df["neg_count"].sum()
    month_df["pos_prop"] = month_df["pos_count"] / (month_df["pos_count"] + month_df["neg_count"])

    print("\n月份分布 (正向响应比例):")
    for _, row in month_df.iterrows():
        month_val = row['month']
        if pd.isna(month_val):
            month_str = "NA"
        else:
            month_str = f"{int(month_val):2d}"
        pos_count = int(row['pos_count'])
        neg_count = int(row['neg_count'])
        print(f"  月份 {month_str}: {pos_count:4d} 正向, {neg_count:4d} 负向, "
              f"正向比例: {row['pos_prop']:.3f}")

    # 纬度分布
    def lat_to_band(lat):
        """将纬度转换为纬度带"""
        if lat < -20:
            return "<-20°S"
        elif lat < -10:
            return "-20°S~-10°S"
        elif lat < 0:
            return "-10°S~0°"
        elif lat < 10:
            return "0°~10°N"
        elif lat < 20:
            return "10°N~20°N"
        else:
            return ">20°N"

    # 确定纬度列名
    lat_col = None
    for col in ['lat_c', 'lat_c_x', 'lat_c_y']:
        if col in pos_samples.columns:
            lat_col = col
            break
    if lat_col is None:
        print("警告: 未找到纬度列，跳过纬度分析")
        lat_col = 'lat_c'  # 使用默认列名，即使不存在

    pos_samples["lat_band"] = pos_samples[lat_col].apply(lat_to_band)
    neg_samples["lat_band"] = neg_samples[lat_col].apply(lat_to_band)

    pos_lat_counts = pos_samples["lat_band"].value_counts()
    neg_lat_counts = neg_samples["lat_band"].value_counts()

    lat_bands = sorted(set(list(pos_lat_counts.index) + list(neg_lat_counts.index)))

    print("\n纬度带分布:")
    for band in lat_bands:
        pos_c = pos_lat_counts.get(band, 0)
        neg_c = neg_lat_counts.get(band, 0)
        total = pos_c + neg_c
        if total > 0:
            pos_prop = pos_c / total
            print(f"  {band:15s}: {pos_c:4d} 正向, {neg_c:4d} 负向, 正向比例: {pos_prop:.3f}")

    # 8. 保存结果
    print("\n[8] 保存结果...")

    # 保存特征比较结果
    results_path = os.path.join(OUTPUT_DIR, "feature_comparison.csv")
    results_df.to_csv(results_path, index=False)
    print(f"  特征比较结果保存至: {results_path}")

    # 保存月份分布
    month_path = os.path.join(OUTPUT_DIR, "monthly_distribution.csv")
    month_df.to_csv(month_path, index=False)
    print(f"  月份分布保存至: {month_path}")

    # 保存样本数据
    pos_path = os.path.join(OUTPUT_DIR, "positive_samples.csv")
    neg_path = os.path.join(OUTPUT_DIR, "negative_samples.csv")
    pos_samples.to_csv(pos_path, index=False)
    neg_samples.to_csv(neg_path, index=False)
    print(f"  正向响应样本保存至: {pos_path}")
    print(f"  负向响应样本保存至: {neg_path}")

    # 9. 生成总结报告
    print("\n[9] 关键发现总结:")
    print("-" * 80)

    # 找出差异最显著的特征
    sig_features = results_df[results_df["p_value"] < 0.05].copy()

    if len(sig_features) > 0:
        print(f"发现 {len(sig_features)} 个显著差异特征 (p < 0.05):")
        for _, row in sig_features.head(10).iterrows():
            direction = "更高" if row["mean_diff"] > 0 else "更低"
            print(f"  • {row['feature']}: 正向响应样本比负向响应样本 {direction} "
                  f"(均值差: {row['mean_diff']:.3f}, p={row['p_value']:.3e})")
    else:
        print("未发现显著差异特征 (p < 0.05)")

    # 月份分析
    max_month = month_df.loc[month_df["pos_prop"].idxmax()]
    min_month = month_df.loc[month_df["pos_prop"].idxmin()]
    print(f"\n月份差异:")
    print(f"  • 正向响应比例最高的月份: {int(max_month['month'])}月 ({max_month['pos_prop']:.3f})")
    print(f"  • 正向响应比例最低的月份: {int(min_month['month'])}月 ({min_month['pos_prop']:.3f})")

    # 纬度分析
    lat_props = []
    for band in lat_bands:
        pos_c = pos_lat_counts.get(band, 0)
        neg_c = neg_lat_counts.get(band, 0)
        total = pos_c + neg_c
        if total > 50:  # 只考虑有足够样本的纬度带
            lat_props.append((band, pos_c / total, total))

    if lat_props:
        lat_props.sort(key=lambda x: x[1], reverse=True)
        print(f"\n纬度带差异 (样本数 > 50):")
        print(f"  • 正向响应比例最高的纬度带: {lat_props[0][0]} ({lat_props[0][1]:.3f}, n={lat_props[0][2]})")
        print(f"  • 正向响应比例最低的纬度带: {lat_props[-1][0]} ({lat_props[-1][1]:.3f}, n={lat_props[-1][2]})")

    print("-" * 80)
    print(f"\n分析完成! 详细结果见: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()