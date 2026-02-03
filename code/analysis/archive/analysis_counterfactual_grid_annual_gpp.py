#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_counterfactual_grid_annual_gpp.py

作用：
- 计算每个 grid、每一年 的年尺度反事实 GPP 变化：
    ΔGPP_year(g,y) = sum_m (GPP_cf - GPP_factual)
- 汇总为 grid 多年统计特征，用于判断：
    年尺度负值是“少数主导”还是“系统性一致”

输出：
- grid_annual_delta_gpp.csv        （grid × year）
- grid_multiyear_summary.csv       （grid 多年统计）
"""

import argparse
import pandas as pd
import numpy as np
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factual", required=True)
    parser.add_argument("--counterfactual", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. 读取数据
    # ------------------------------------------------------------------
    df_f = pd.read_csv(args.factual)
    df_c = pd.read_csv(args.counterfactual)

    key_cols = ["grid_id", "year", "month"]
    for c in key_cols:
        if c not in df_f.columns or c not in df_c.columns:
            raise ValueError(f"缺少必要列: {c}")

    # ------------------------------------------------------------------
    # 2. 合并 factual / counterfactual
    # ------------------------------------------------------------------
    df = (
        df_f[key_cols + ["gpp_pred"]]
        .rename(columns={"gpp_pred": "gpp_factual"})
        .merge(
            df_c[key_cols + ["gpp_pred"]]
            .rename(columns={"gpp_pred": "gpp_cf"}),
            on=key_cols,
            how="inner",
        )
    )

    # 月差
    df["delta_gpp_month"] = df["gpp_cf"] - df["gpp_factual"]

    # ------------------------------------------------------------------
    # 3. grid × year 年尺度汇总
    # ------------------------------------------------------------------
    grid_year = (
        df.groupby(["grid_id", "year"])
        .agg(
            delta_gpp_year=("delta_gpp_month", "sum"),
            n_months=("delta_gpp_month", "count"),
        )
        .reset_index()
    )

    grid_year_path = os.path.join(args.outdir, "grid_annual_delta_gpp.csv")
    grid_year.to_csv(grid_year_path, index=False)

    # ------------------------------------------------------------------
    # 4. grid 多年统计
    # ------------------------------------------------------------------
    summary = (
        grid_year
        .groupby("grid_id")
        .agg(
            n_years=("year", "count"),
            n_year_pos=("delta_gpp_year", lambda x: (x > 0).sum()),
            n_year_neg=("delta_gpp_year", lambda x: (x < 0).sum()),
            mean_delta_gpp_year=("delta_gpp_year", "mean"),
            median_delta_gpp_year=("delta_gpp_year", "median"),
        )
        .reset_index()
    )

    summary["pos_ratio"] = summary["n_year_pos"] / summary["n_years"]

    summary_path = os.path.join(args.outdir, "grid_multiyear_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n[OK] Grid 年尺度反事实 GPP 分解完成")
    print("输出文件：")
    print(" -", grid_year_path)
    print(" -", summary_path)

    # 简单预览
    print("\n[Preview] Top 10 negative grids (by mean ΔGPP_year):")
    print(
        summary.sort_values("mean_delta_gpp_year")
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
