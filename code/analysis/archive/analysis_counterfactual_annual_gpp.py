#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_counterfactual_annual_gpp.py

作用：
- 计算年尺度反事实差异，并同时给出两种 delta（避免概念混淆）：

(1) delta_cf_minus_true:
    = gpp_pred_counterfactual - gpp_true
    这是你现在想要的“反事实预测 vs 实际观测”的差

(2) delta_cf_minus_factual:
    = gpp_pred_counterfactual - gpp_pred_factual
    这是“移除 MHW 在模型内部造成的变化”

输入：
- factual_rolling_predictions.csv: 必须含 grid_id/year/month/gpp_pred/gpp_true
- counterfactual_rolling_predictions.csv: 必须含 grid_id/year/month/gpp_pred
输出：
- --out 指定的 annual 汇总表（按 year 聚合）
"""

import argparse
import os
import pandas as pd


KEY_COLS = ["grid_id", "year", "month"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--factual", required=True)
    p.add_argument("--counterfactual", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    df_f = pd.read_csv(args.factual)
    df_c = pd.read_csv(args.counterfactual)

    # --- 必要列检查 ---
    for k in KEY_COLS:
        if k not in df_f.columns:
            raise ValueError(f"[factual] 缺少键列: {k}")
        if k not in df_c.columns:
            raise ValueError(f"[counterfactual] 缺少键列: {k}")

    if "gpp_pred" not in df_f.columns:
        raise ValueError("[factual] 缺少列: gpp_pred")
    if "gpp_true" not in df_f.columns:
        raise ValueError("[factual] 缺少列: gpp_true")
    if "gpp_pred" not in df_c.columns:
        raise ValueError("[counterfactual] 缺少列: gpp_pred")

    # --- 合并对齐（按 grid_id/year/month） ---
    df = (
        df_f[KEY_COLS + ["gpp_true", "gpp_pred"]]
        .rename(columns={"gpp_pred": "gpp_pred_factual"})
        .merge(
            df_c[KEY_COLS + ["gpp_pred"]].rename(columns={"gpp_pred": "gpp_pred_cf"}),
            on=KEY_COLS,
            how="inner",
            validate="one_to_one",
        )
    )

    # --- 两种 delta（月尺度） ---
    df["delta_cf_minus_true_month"] = df["gpp_pred_cf"] - df["gpp_true"]
    df["delta_cf_minus_factual_month"] = df["gpp_pred_cf"] - df["gpp_pred_factual"]

    # --- 年尺度汇总（全局：所有 grid 的年总和） ---
    annual = (
        df.groupby("year", as_index=False)
        .agg(
            delta_cf_minus_true_year=("delta_cf_minus_true_month", "sum"),
            delta_cf_minus_factual_year=("delta_cf_minus_factual_month", "sum"),
            mean_delta_cf_minus_true_month=("delta_cf_minus_true_month", "mean"),
            mean_delta_cf_minus_factual_month=("delta_cf_minus_factual_month", "mean"),
            n_months=("delta_cf_minus_true_month", "count"),
            n_grids=("grid_id", "nunique"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )

    # 方向标签（按你关心的 cf - true）
    annual["direction_cf_minus_true"] = annual["delta_cf_minus_true_year"].apply(
        lambda x: "increase" if x > 0 else ("decrease" if x < 0 else "neutral")
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    annual.to_csv(args.out, index=False)

    print("\n[OK] 年尺度统计完成（同时输出 cf-true 与 cf-factual 两种 delta）")
    print(annual)


if __name__ == "__main__":
    main()
