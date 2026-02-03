# ============================================================
# 年度GPP损失统计
#
# 统计三种类型的年度GPP变化：
# 1. 预测误差损失：GPP预测值 vs GPP真实值
# 2. MHW导致的GPP损失：事实预测 vs 反事实预测
# 3. GPP真实值的年度变化
#
# 输出：
#   - CSV表格：每年的汇总统计
#   - 可视化图表：年度趋势
# ============================================================

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--factual-csv",
        type=str,
        default="results/predictions/factual_rolling_predictions.csv",
        help="事实预测结果CSV",
    )
    parser.add_argument(
        "--counterfactual-csv",
        type=str,
        default="results/predictions/counterfactual_rolling_predictions.csv",
        help="反事实预测结果CSV",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/statistics",
        help="输出目录",
    )
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)

    print("[INFO] 加载数据...")

    # 加载数据
    df_factual = pd.read_csv(args.factual_csv)
    df_counter = pd.read_csv(args.counterfactual_csv)

    # 重命名列以避免混淆
    df_factual = df_factual.rename(columns={"gpp_pred": "gpp_pred_factual"})
    df_counter = df_counter.rename(columns={"gpp_pred": "gpp_pred_counterfactual"})

    # 合并数据
    df = pd.merge(
        df_factual[["grid_id", "year", "month", "lat_c", "gpp_true", "gpp_pred_factual"]],
        df_counter[["grid_id", "year", "month", "gpp_pred_counterfactual"]],
        on=["grid_id", "year", "month"],
        how="inner"
    )

    # 确保year是整数类型
    df["year"] = df["year"].astype(int)

    print(f"[INFO] 合并后数据行数: {len(df)}")
    print(f"[INFO] 年份范围: {df['year'].min()} - {df['year'].max()}")

    # ============================================================
    # 1. 计算各种ΔGPP
    # ============================================================

    # 1.1 预测误差（事实预测 vs 真实值）
    df["delta_gpp_pred_error"] = df["gpp_pred_factual"] - df["gpp_true"]
    df["delta_gpp_pred_error_pct"] = (df["delta_gpp_pred_error"] / df["gpp_true"]) * 100

    # 1.2 MHW导致的GPP变化（事实预测 vs 反事实预测）
    df["delta_gpp_mhw"] = df["gpp_pred_factual"] - df["gpp_pred_counterfactual"]
    df["delta_gpp_mhw_pct"] = (df["delta_gpp_mhw"] / df["gpp_pred_counterfactual"]) * 100

    # 1.3 MHW影响（另一种定义：事实预测与反事实预测的相对差异）
    df["mhw_impact_ratio"] = df["gpp_pred_factual"] / df["gpp_pred_counterfactual"]

    # ============================================================
    # 2. 年度汇总统计
    # ============================================================

    # 按年分组
    annual_stats = []

    for year in sorted(df["year"].unique()):
        df_year = df[df["year"] == year]

        stats = {
            "year": year,
            "n_grid_months": len(df_year),
            "n_grids": df_year["grid_id"].nunique(),
        }

        # GPP真实值统计
        stats["gpp_true_sum"] = df_year["gpp_true"].sum()
        stats["gpp_true_mean"] = df_year["gpp_true"].mean()
        stats["gpp_true_median"] = df_year["gpp_true"].median()

        # 预测误差统计
        stats["pred_error_mean"] = df_year["delta_gpp_pred_error"].mean()
        stats["pred_error_median"] = df_year["delta_gpp_pred_error"].median()
        stats["pred_error_sum"] = df_year["delta_gpp_pred_error"].sum()
        stats["pred_error_pct_mean"] = df_year["delta_gpp_pred_error_pct"].mean()
        stats["pred_error_pct_median"] = df_year["delta_gpp_pred_error_pct"].median()

        # MHW影响统计
        stats["mhw_effect_mean"] = df_year["delta_gpp_mhw"].mean()
        stats["mhw_effect_median"] = df_year["delta_gpp_mhw"].median()
        stats["mhw_effect_sum"] = df_year["delta_gpp_mhw"].sum()
        stats["mhw_effect_pct_mean"] = df_year["delta_gpp_mhw_pct"].mean()
        stats["mhw_effect_pct_median"] = df_year["delta_gpp_mhw_pct"].median()

        # MHW影响比例统计
        stats["mhw_impact_ratio_mean"] = df_year["mhw_impact_ratio"].mean()
        stats["mhw_impact_ratio_median"] = df_year["mhw_impact_ratio"].median()

        # 正值/负值比例
        stats["pred_error_positive_pct"] = (df_year["delta_gpp_pred_error"] > 0).mean() * 100
        stats["pred_error_negative_pct"] = (df_year["delta_gpp_pred_error"] < 0).mean() * 100
        stats["mhw_effect_positive_pct"] = (df_year["delta_gpp_mhw"] > 0).mean() * 100
        stats["mhw_effect_negative_pct"] = (df_year["delta_gpp_mhw"] < 0).mean() * 100

        annual_stats.append(stats)

    # 创建DataFrame
    df_annual = pd.DataFrame(annual_stats)

    # ============================================================
    # 3. 保存统计表格
    # ============================================================

    # 详细统计表
    detailed_path = os.path.join(args.out_dir, "tables", "annual_gpp_statistics_detailed.csv")
    df_annual.to_csv(detailed_path, index=False, float_format="%.4f")
    print(f"[INFO] 详细统计表保存至: {detailed_path}")

    # 简化统计表（关键指标）
    key_cols = [
        "year", "n_grid_months", "n_grids",
        "gpp_true_sum", "gpp_true_mean",
        "pred_error_sum", "pred_error_pct_median",
        "mhw_effect_sum", "mhw_effect_pct_median",
        "mhw_impact_ratio_median"
    ]
    df_key = df_annual[key_cols].copy()

    # 重命名列以便阅读
    rename_dict = {
        "year": "年份",
        "n_grid_months": "网格-月份数",
        "n_grids": "网格数",
        "gpp_true_sum": "GPP真实值总和",
        "gpp_true_mean": "GPP真实值均值",
        "pred_error_sum": "预测误差总和",
        "pred_error_pct_median": "预测误差百分比中位数",
        "mhw_effect_sum": "MHW影响总和",
        "mhw_effect_pct_median": "MHW影响百分比中位数",
        "mhw_impact_ratio_median": "MHW影响比例中位数"
    }
    df_key = df_key.rename(columns=rename_dict)

    key_path = os.path.join(args.out_dir, "tables", "annual_gpp_statistics_key.csv")
    df_key.to_csv(key_path, index=False, float_format="%.4f")
    print(f"[INFO] 关键统计表保存至: {key_path}")

    # ============================================================
    # 4. 生成可视化图表
    # ============================================================

    fig_dir = os.path.join(args.out_dir, "figures")

    # 4.1 GPP真实值年度趋势
    plt.figure(figsize=(10, 6))
    plt.bar(df_annual["year"], df_annual["gpp_true_sum"] / 1e9, color="skyblue")
    plt.xlabel("年份")
    plt.ylabel("GPP真实值总和 (10^9)")
    plt.title("年度GPP真实值总和")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "annual_gpp_true_sum.png"), dpi=300)
    plt.close()

    # 4.2 预测误差年度趋势
    plt.figure(figsize=(10, 6))
    plt.bar(df_annual["year"], df_annual["pred_error_sum"] / 1e9,
            color="lightcoral", label="预测误差总和")
    plt.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    plt.xlabel("年份")
    plt.ylabel("预测误差总和 (10^9)")
    plt.title("年度预测误差总和（事实预测 - 真实值）")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "annual_prediction_error_sum.png"), dpi=300)
    plt.close()

    # 4.3 MHW影响年度趋势
    plt.figure(figsize=(10, 6))
    plt.bar(df_annual["year"], df_annual["mhw_effect_sum"] / 1e9,
            color="lightgreen", label="MHW影响总和")
    plt.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    plt.xlabel("年份")
    plt.ylabel("MHW影响总和 (10^9)")
    plt.title("年度MHW影响总和（事实预测 - 反事实预测）")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "annual_mhw_effect_sum.png"), dpi=300)
    plt.close()

    # 4.4 MHW影响比例中位数年度趋势
    plt.figure(figsize=(10, 6))
    plt.plot(df_annual["year"], df_annual["mhw_impact_ratio_median"],
             marker="o", linestyle="-", linewidth=2, color="purple")
    plt.axhline(y=1.0, color="red", linestyle="--", linewidth=1, label="基准线 (1.0)")
    plt.xlabel("年份")
    plt.ylabel("MHW影响比例中位数")
    plt.title("年度MHW影响比例中位数（事实预测 / 反事实预测）")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "annual_mhw_impact_ratio_median.png"), dpi=300)
    plt.close()

    # 4.5 综合趋势图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 子图1: GPP真实值
    axes[0, 0].bar(df_annual["year"], df_annual["gpp_true_sum"] / 1e9, color="skyblue")
    axes[0, 0].set_xlabel("年份")
    axes[0, 0].set_ylabel("GPP真实值总和 (10^9)")
    axes[0, 0].set_title("(a) 年度GPP真实值总和")
    axes[0, 0].grid(True, alpha=0.3)

    # 子图2: 预测误差百分比中位数
    axes[0, 1].bar(df_annual["year"], df_annual["pred_error_pct_median"], color="lightcoral")
    axes[0, 1].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    axes[0, 1].set_xlabel("年份")
    axes[0, 1].set_ylabel("预测误差百分比中位数 (%)")
    axes[0, 1].set_title("(b) 年度预测误差百分比中位数")
    axes[0, 1].grid(True, alpha=0.3)

    # 子图3: MHW影响百分比中位数
    axes[1, 0].bar(df_annual["year"], df_annual["mhw_effect_pct_median"], color="lightgreen")
    axes[1, 0].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    axes[1, 0].set_xlabel("年份")
    axes[1, 0].set_ylabel("MHW影响百分比中位数 (%)")
    axes[1, 0].set_title("(c) 年度MHW影响百分比中位数")
    axes[1, 0].grid(True, alpha=0.3)

    # 子图4: MHW影响比例中位数
    axes[1, 1].plot(df_annual["year"], df_annual["mhw_impact_ratio_median"],
                   marker="o", linestyle="-", linewidth=2, color="purple")
    axes[1, 1].axhline(y=1.0, color="red", linestyle="--", linewidth=1, label="基准线 (1.0)")
    axes[1, 1].set_xlabel("年份")
    axes[1, 1].set_ylabel("MHW影响比例中位数")
    axes[1, 1].set_title("(d) 年度MHW影响比例中位数")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "annual_gpp_trends_comprehensive.png"), dpi=300)
    plt.close()

    print(f"[INFO] 图表保存至: {fig_dir}")

    # ============================================================
    # 5. 输出汇总统计
    # ============================================================

    print("\n" + "="*60)
    print("年度GPP损失统计汇总")
    print("="*60)

    # 整体统计
    print("\n=== 整体统计 (所有年份) ===")
    print(f"总网格-月份数: {len(df)}")
    print(f"总网格数: {df['grid_id'].nunique()}")
    print(f"GPP真实值总和: {df['gpp_true'].sum() / 1e9:.2f} × 10^9")
    print(f"预测误差总和: {df['delta_gpp_pred_error'].sum() / 1e9:.2f} × 10^9")
    print(f"MHW影响总和: {df['delta_gpp_mhw'].sum() / 1e9:.2f} × 10^9")

    print("\n=== 预测误差统计 ===")
    print(f"预测误差中位数: {df['delta_gpp_pred_error'].median() / 1e6:.2f} × 10^6")
    print(f"预测误差百分比中位数: {df['delta_gpp_pred_error_pct'].median():.2f}%")
    print(f"高估比例 (ΔGPP > 0): {(df['delta_gpp_pred_error'] > 0).mean() * 100:.1f}%")
    print(f"低估比例 (ΔGPP < 0): {(df['delta_gpp_pred_error'] < 0).mean() * 100:.1f}%")

    print("\n=== MHW影响统计 ===")
    print(f"MHW影响中位数: {df['delta_gpp_mhw'].median() / 1e6:.2f} × 10^6")
    print(f"MHW影响百分比中位数: {df['delta_gpp_mhw_pct'].median():.2f}%")
    print(f"促进比例 (ΔGPP_MHW > 0): {(df['delta_gpp_mhw'] > 0).mean() * 100:.1f}%")
    print(f"抑制比例 (ΔGPP_MHW < 0): {(df['delta_gpp_mhw'] < 0).mean() * 100:.1f}%")
    print(f"MHW影响比例中位数: {df['mhw_impact_ratio'].median():.4f}")

    print("\n=== 年度关键指标 ===")
    print(df_key.to_string(index=False))

    print(f"\n[INFO] 统计完成！")
    print(f"详细结果见: {args.out_dir}")

if __name__ == "__main__":
    main()