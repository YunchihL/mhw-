#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_step5_event_month_contribution.py

Step 5｜事件月分析
- 比较 MHW 月份 vs 非 MHW 月份 的 GPP impact（cf - factual）
- 回答：
  1) 事件月的 impact 是否更负？
  2) 事件月负值比例是否更高？
  3) 年尺度总损失中，事件月贡献了多少？
"""

import argparse
import os
import pandas as pd


KEY_COLS = ["grid_id", "year", "month"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--factual", required=True)
    p.add_argument("--counterfactual", required=True)
    p.add_argument("--data", required=True, help="data/data.csv")
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    df_f = pd.read_csv(args.factual)
    df_c = pd.read_csv(args.counterfactual)
    df_d = pd.read_csv(args.data)

    # 检查列
    for c in KEY_COLS + ["gpp_pred"]:
        if c not in df_f.columns or c not in df_c.columns:
            raise ValueError(f"预测文件缺少列: {c}")
    if "isMHW" not in df_d.columns:
        raise ValueError("data.csv 缺少列: isMHW")

    # 合并预测
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

    # 合并 isMHW
    df = df.merge(
        df_d[KEY_COLS + ["isMHW"]],
        on=KEY_COLS,
        how="left",
    )

    # 分组统计
    summary = (
        df.groupby("isMHW")
        .agg(
            n=("impact", "count"),
            mean_impact=("impact", "mean"),
            median_impact=("impact", "median"),
            neg_ratio=("impact", lambda x: (x < 0).mean()),
            total_impact=("impact", "sum"),
        )
        .reset_index()
    )

    # 年尺度总损失
    total_loss = df["impact"].sum()

    summary["share_of_total"] = summary["total_impact"] / total_loss

    # 输出
    os.makedirs(args.outdir, exist_ok=True)
    out_csv = os.path.join(args.outdir, "event_month_vs_non_event_summary.csv")
    summary.to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print("[STEP 5 OK] Event-month contribution analysis")
    print("=" * 80)
    print(summary.to_string(index=False))
    print("-" * 80)
    print(f"Total impact (all months) = {total_loss:.6e}")
    print("=" * 80)
    print("[NOTE]")
    print("isMHW = 1 → event months")
    print("isMHW = 0 → non-event months")
    print("=" * 80)


if __name__ == "__main__":
    main()
