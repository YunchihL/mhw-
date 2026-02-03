#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_answer_three_key_questions.py

最终对齐版：
- 使用你 data.csv 中真实存在的 MHW 变量：
    - duration_weighted_sum
    - intensity_cumulative_weighted_sum
    - intensity_density
- Q1：lat_c + lon_c
- Q2：背景 GPP（factual 的 gpp_true）
- Q3：MHW 热灾害结构（年尺度 → 多年平均）

不输出大表，只给“结果级”结论。
"""

import argparse
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--factual", required=True)
    parser.add_argument(
        "--data",
        default="/home/linyunzhi/project_for_journal/data/data.csv",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------
    # 1. 读取数据
    # ------------------------------------------------------------
    summary = pd.read_csv(args.summary)
    factual = pd.read_csv(args.factual)
    data = pd.read_csv(args.data)

    # ------------------------------------------------------------
    # 2. NEG / OTHERS 分组
    # ------------------------------------------------------------
    summary["group"] = np.where(
        summary["n_year_neg"] > summary["n_year_pos"],
        "NEG",
        "OTHERS",
    )
    grid_group = summary[["grid_id", "group"]]

    # ------------------------------------------------------------
    # 3. 背景 GPP（grid 多年平均）
    # ------------------------------------------------------------
    if "grid_id" not in factual.columns or "gpp_true" not in factual.columns:
        raise ValueError("factual 中必须包含 grid_id 和 gpp_true")

    grid_gpp = (
        factual
        .groupby("grid_id")["gpp_true"]
        .mean()
        .reset_index()
    )

    # ------------------------------------------------------------
    # 4. MHW 热灾害指标（grid × year → grid）
    # ------------------------------------------------------------
    mhw_cols = [
        "grid_id", "year", "lat_c", "lon_c",
        "duration_weighted_sum",
        "intensity_cumulative_weighted_sum",
        "intensity_density",
    ]
    for c in mhw_cols:
        if c not in data.columns:
            raise ValueError(f"data.csv 中缺少必要列: {c}")

    # 年尺度汇总
    grid_year = (
        data
        .groupby(["grid_id", "year"])
        .agg(
            lat=("lat_c", "mean"),
            lon=("lon_c", "mean"),
            duration_year=("duration_weighted_sum", "sum"),
            intensity_year=("intensity_cumulative_weighted_sum", "sum"),
            intensity_density_year=("intensity_density", "mean"),
        )
        .reset_index()
    )

    # 多年平均 → grid
    grid_mhw = (
        grid_year
        .groupby("grid_id")
        .agg(
            lat=("lat", "mean"),
            lon=("lon", "mean"),
            duration=("duration_year", "mean"),
            intensity=("intensity_year", "mean"),
            intensity_density=("intensity_density_year", "mean"),
        )
        .reset_index()
    )

    # ------------------------------------------------------------
    # 5. 合并所有信息
    # ------------------------------------------------------------
    grid_all = (
        grid_group
        .merge(grid_gpp, on="grid_id")
        .merge(grid_mhw, on="grid_id")
    )

    # ------------------------------------------------------------
    # 6. 组间统计
    # ------------------------------------------------------------
    stats = (
        grid_all
        .groupby("group")
        .agg(
            lat_mean=("lat", "mean"),
            lon_mean=("lon", "mean"),
            gpp_median=("gpp_true", "median"),
            dur_mean=("duration", "mean"),
            inten_mean=("intensity", "mean"),
            iden_mean=("intensity_density", "mean"),
            n_grids=("grid_id", "count"),
        )
    )

    # ------------------------------------------------------------
    # 7. 输出结果级结论
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("三 个 关 键 问 题（完全对齐 data.csv 变量定义）")
    print("=" * 80)

    # Q1 空间
    print("\n[Q1] 空间分布（多年平均）")
    print(stats[["lat_mean", "lon_mean"]])
    print(
        f"→ 纬度差 (NEG - OTHERS) = "
        f"{stats.loc['NEG','lat_mean'] - stats.loc['OTHERS','lat_mean']:.2f}"
    )
    print(
        f"→ 经度差 (NEG - OTHERS) = "
        f"{stats.loc['NEG','lon_mean'] - stats.loc['OTHERS','lon_mean']:.2f}"
    )

    # Q2 背景 GPP
    print("\n[Q2] 背景 GPP（中位数）")
    print(stats[["gpp_median"]])

    # Q3 MHW 热灾害结构
    print("\n[Q3] MHW 热灾害结构（多年平均）")
    print(stats[["dur_mean", "inten_mean", "iden_mean"]])

    print("\n变量说明：")
    print("- duration_weighted_sum → 时间尺度（持续性）")
    print("- intensity_cumulative_weighted_sum → 累积热量")
    print("- intensity_density → 热灾害速度（强而短 vs 弱而长）")

    print("\n" + "=" * 80)
    print("[OK] 分析完成（变量定义完全一致）")
    print("=" * 80)


if __name__ == "__main__":
    main()
