#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_step2_spatial_concentration.py

Step 2：空间异质性（grid 贡献集中度）

- 基于 cf - factual（月尺度 impact）
- 在 grid × year 尺度求和
- 再对每个 grid 计算多年平均年 impact
- 评估：
  * 负贡献 grid 占比
  * Top N grid 对总损失的贡献比例
"""

import argparse
import os
import pandas as pd


KEY_COLS = ["grid_id", "year", "month"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--factual", required=True)
    p.add_argument("--counterfactual", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--topn", type=int, default=10, help="Top N grids (default=10)")
    args = p.parse_args()

    df_f = pd.read_csv(args.factual)
    df_c = pd.read_csv(args.counterfactual)

    for c in KEY_COLS + ["gpp_pred"]:
        if c not in df_f.columns:
            raise ValueError(f"[factual] 缺少列: {c}")
        if c not in df_c.columns:
            raise ValueError(f"[counterfactual] 缺少列: {c}")

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

    df["impact"] = df["gpp_pred_cf"] - df["gpp_pred_factual"]

    # grid-year 年总 impact
    gy = (
        df.groupby(["grid_id", "year"], as_index=False)
        .agg(impact_year=("impact", "sum"))
    )

    # grid 多年平均年 impact
    gmean = (
        gy.groupby("grid_id", as_index=False)
        .agg(
            mean_impact_year=("impact_year", "mean"),
            n_years=("year", "count"),
        )
    )

    # 只看负贡献（损失）
    neg = gmean[gmean["mean_impact_year"] < 0].copy()
    neg["abs_impact"] = -neg["mean_impact_year"]

    total_loss = neg["abs_impact"].sum()

    neg = neg.sort_values("abs_impact", ascending=False).reset_index(drop=True)
    neg["cum_share"] = neg["abs_impact"].cumsum() / total_loss

    topn = min(args.topn, neg.shape[0])
    topn_share = neg.loc[topn - 1, "cum_share"]

    # 输出
    os.makedirs(args.outdir, exist_ok=True)
    out_csv = os.path.join(args.outdir, "grid_mean_impact_cf_minus_factual.csv")
    out_txt = os.path.join(args.outdir, "spatial_concentration_summary.txt")

    gmean.to_csv(out_csv, index=False)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("Step 2 | Spatial concentration of MHW-associated GPP loss\n")
        f.write("=" * 72 + "\n")
        f.write(f"Total grids: {gmean.shape[0]}\n")
        f.write(f"Negative-impact grids: {neg.shape[0]} ({neg.shape[0]/gmean.shape[0]:.2%})\n")
        f.write("\n")
        f.write(f"Total mean annual loss (sum over grids) = {total_loss:.6e}\n")
        f.write(f"Top {topn} grids share of total loss = {topn_share:.2%}\n")
        f.write("\n")
        f.write("Top grids (preview):\n")
        f.write(neg.head(topn).to_string(index=False))
        f.write("\n")

    # 终端简要输出
    print("\n" + "=" * 80)
    print("[STEP 2 OK] Spatial concentration computed")
    print(f"Negative-impact grids: {neg.shape[0]} / {gmean.shape[0]}")
    print(f"Top {topn} grids share of total loss: {topn_share:.2%}")
    print("=" * 80)


if __name__ == "__main__":
    main()
