#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_step4_impact_vs_mhw_structure.py

Step 4：
- 在 grid × year 尺度上
- 计算 MHW-associated GPP impact（cf - factual）
- 与 MHW 结构变量的关系（相关系数 + 分位数分箱）

不做因果、不建复杂模型，只做描述性关系。
"""

import argparse
import os
import pandas as pd
import numpy as np


KEY_COLS = ["grid_id", "year", "month"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--factual", required=True)
    p.add_argument("--counterfactual", required=True)
    p.add_argument("--data", required=True, help="原始 data.csv")
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    df_f = pd.read_csv(args.factual)
    df_c = pd.read_csv(args.counterfactual)
    df_d = pd.read_csv(args.data)

    # 检查必要列
    for c in KEY_COLS + ["gpp_pred"]:
        if c not in df_f.columns or c not in df_c.columns:
            raise ValueError(f"预测文件缺少列: {c}")

    mhw_cols = [
        "duration_weighted_sum",
        "intensity_cumulative_weighted_sum",
        "intensity_density",
        "isMHW",
    ]
    for c in mhw_cols:
        if c not in df_d.columns:
            raise ValueError(f"data.csv 缺少列: {c}")

    # 合并预测，算 impact（月）
    df = (
        df_f[KEY_COLS + ["gpp_pred"]]
        .rename(columns={"gpp_pred": "gpp_pred_factual"})
        .merge(
            df_c[KEY_COLS + ["gpp_pred"]].rename(columns={"gpp_pred": "gpp_pred_cf"}),
            on=KEY_COLS,
            how="inner",
        )
    )
    df["impact"] = df["gpp_pred_cf"] - df["gpp_pred_factual"]

    # 合并 MHW 变量
    df = df.merge(
        df_d[KEY_COLS + mhw_cols],
        on=KEY_COLS,
        how="left",
    )

    # 仅在 MHW 月份聚合 MHW 结构
    df_mhw = df[df["isMHW"] == 1].copy()

    # grid × year 聚合
    gy = (
        df.groupby(["grid_id", "year"], as_index=False)
        .agg(
            impact_year=("impact", "sum"),
        )
    )

    mhw_gy = (
        df_mhw.groupby(["grid_id", "year"], as_index=False)
        .agg(
            duration_year=("duration_weighted_sum", "mean"),
            intensity_year=("intensity_cumulative_weighted_sum", "mean"),
            density_year=("intensity_density", "mean"),
            n_mhw_months=("isMHW", "count"),
        )
    )

    gy = gy.merge(mhw_gy, on=["grid_id", "year"], how="left")

    # 相关性（Spearman 更稳健）
    corr = gy[
        ["impact_year", "duration_year", "intensity_year", "density_year"]
    ].corr(method="spearman")

    # 输出
    os.makedirs(args.outdir, exist_ok=True)
    gy.to_csv(os.path.join(args.outdir, "grid_year_impact_and_mhw_structure.csv"), index=False)
    corr.to_csv(os.path.join(args.outdir, "impact_vs_mhw_structure_spearman.csv"))

    print("\n" + "=" * 80)
    print("[STEP 4 OK] Impact vs MHW structure computed")
    print("Spearman correlations:")
    print(corr)
    print("=" * 80)


if __name__ == "__main__":
    main()
