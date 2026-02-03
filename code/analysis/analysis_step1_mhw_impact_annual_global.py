#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_step1_mhw_impact_annual_global.py

Step 1（主结果定型）：
- 用“模型内效应”定义 MHW 影响：
    impact = gpp_pred_cf - gpp_pred_factual
  （impact < 0 表示：移除 MHW 会让预测更低；impact > 0 表示：移除 MHW 让预测更高）
- 年尺度：对每年所有 grid×month 的 impact 求和
- 输出：
  1) 每年 impact_year（global sum）
  2) 全期累计 impact_total
  3) 年份层面的正/负占比
  4) 简单的“净损失”口径：loss_total = -impact_total

写出：
- results/annual_global_mhw_impact_cf_minus_factual.csv
- results/annual_global_mhw_impact_cf_minus_factual_summary.txt
"""

import argparse
import os
import pandas as pd


KEY_COLS = ["grid_id", "year", "month"]


def must_have_cols(df: pd.DataFrame, cols, name: str):
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"[{name}] 缺少必要列: {c}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--factual", required=True, help="factual_rolling_predictions.csv")
    p.add_argument("--counterfactual", required=True, help="counterfactual_rolling_predictions.csv")
    p.add_argument("--outdir", required=True, help="输出目录，例如 code/analysis/results")
    args = p.parse_args()

    df_f = pd.read_csv(args.factual)
    df_c = pd.read_csv(args.counterfactual)

    must_have_cols(df_f, KEY_COLS + ["gpp_pred"], "factual")
    must_have_cols(df_c, KEY_COLS + ["gpp_pred"], "counterfactual")

    # 对齐合并
    df = (
        df_f[KEY_COLS + ["gpp_pred"]]
        .rename(columns={"gpp_pred": "gpp_pred_factual"})
        .merge(
            df_c[KEY_COLS + ["gpp_pred"]].rename(columns={"gpp_pred": "gpp_pred_cf"}),
            on=KEY_COLS,
            how="inner",
            validate="one_to_one",
        )
    )

    # 模型内效应（月尺度）
    df["impact_cf_minus_factual_month"] = df["gpp_pred_cf"] - df["gpp_pred_factual"]

    # 年尺度全局汇总
    annual = (
        df.groupby("year", as_index=False)
        .agg(
            impact_year=("impact_cf_minus_factual_month", "sum"),
            mean_impact_month=("impact_cf_minus_factual_month", "mean"),
            n_months=("impact_cf_minus_factual_month", "count"),
            n_grids=("grid_id", "nunique"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )

    annual["direction"] = annual["impact_year"].apply(
        lambda x: "increase" if x > 0 else ("decrease" if x < 0 else "neutral")
    )

    # 全期累计（global）
    impact_total = float(annual["impact_year"].sum())
    # 若你要“损失”为正数口径：loss = -impact
    loss_total = -impact_total

    n_years = int(annual.shape[0])
    n_pos = int((annual["impact_year"] > 0).sum())
    n_neg = int((annual["impact_year"] < 0).sum())
    n_zero = int((annual["impact_year"] == 0).sum())

    # 输出
    os.makedirs(args.outdir, exist_ok=True)
    out_csv = os.path.join(args.outdir, "annual_global_mhw_impact_cf_minus_factual.csv")
    out_txt = os.path.join(args.outdir, "annual_global_mhw_impact_cf_minus_factual_summary.txt")

    annual.to_csv(out_csv, index=False)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("Step 1 | Global annual MHW impact (cf - factual)\n")
        f.write("=" * 72 + "\n")
        f.write(f"Years covered: {n_years}\n")
        f.write(f"Impact definition: impact = gpp_pred_cf - gpp_pred_factual\n")
        f.write("\n")
        f.write("[Annual sign counts]\n")
        f.write(f"  impact_year > 0 (increase): {n_pos} / {n_years} ({n_pos/n_years:.2%})\n")
        f.write(f"  impact_year < 0 (decrease): {n_neg} / {n_years} ({n_neg/n_years:.2%})\n")
        f.write(f"  impact_year = 0 (neutral) : {n_zero} / {n_years} ({n_zero/n_years:.2%})\n")
        f.write("\n")
        f.write("[Cumulative impact across all years]\n")
        f.write(f"  impact_total = {impact_total:.6e}\n")
        f.write("\n")
        f.write("[Loss-style reporting (loss = -impact)]\n")
        f.write(f"  loss_total = {loss_total:.6e}\n")
        f.write("\n")
        f.write("Notes:\n")
        f.write("- impact_total < 0 means cf < factual on average (removing MHW lowers predicted GPP).\n")
        f.write("- loss_total > 0 is only meaningful if you interpret (cf - factual) as MHW-associated effect.\n")

    # 终端只输出“最关键的几行”
    print("\n" + "=" * 80)
    print("[STEP 1 OK] Global annual MHW impact computed (cf - factual)")
    print(f"[OUT] {out_csv}")
    print(f"[OUT] {out_txt}")
    print("=" * 80)
    print(annual[["year", "impact_year", "direction", "n_months", "n_grids"]].to_string(index=False))
    print("-" * 80)
    print(f"impact_total = {impact_total:.6e}")
    print(f"loss_total   = {loss_total:.6e}  (loss = -impact)")
    print("=" * 80)


if __name__ == "__main__":
    main()
