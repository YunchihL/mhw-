#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_step5p_event_month_direction_distribution.py

Step 5p｜MHW 当月 GPP 响应的方向与分布

- 仅分析 isMHW == 1 的月份
- 定义：
    delta_gpp = gpp_pred_cf - gpp_pred_factual
- 统计：
    1) delta_gpp > 0（增加） vs < 0（减少）的比例
    2) delta_gpp 的分布统计（mean / median / quantiles）
    3) |delta_gpp| 的分布（幅度，不看方向）
- 输出：
    - step5p_event_month_direction_summary.csv
    - step5p_event_month_delta_gpp_distribution.csv
"""

import argparse
import os
import pandas as pd
import numpy as np


KEY_COLS = ["grid_id", "year", "month"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factual", required=True,
                        help="factual_rolling_predictions.csv")
    parser.add_argument("--counterfactual", required=True,
                        help="counterfactual_rolling_predictions.csv")
    parser.add_argument("--data", required=True,
                        help="data.csv (must include isMHW)")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------
    df_f = pd.read_csv(args.factual)
    df_c = pd.read_csv(args.counterfactual)
    df_d = pd.read_csv(args.data)

    # sanity check
    for c in KEY_COLS + ["gpp_pred"]:
        if c not in df_f.columns or c not in df_c.columns:
            raise ValueError(f"预测文件缺少列: {c}")
    if "isMHW" not in df_d.columns:
        raise ValueError("data.csv 缺少列: isMHW")

    # ------------------------------------------------------------
    # Merge & compute delta_gpp
    # ------------------------------------------------------------
    df = (
        df_f[KEY_COLS + ["gpp_pred"]]
        .rename(columns={"gpp_pred": "gpp_factual"})
        .merge(
            df_c[KEY_COLS + ["gpp_pred"]]
            .rename(columns={"gpp_pred": "gpp_cf"}),
            on=KEY_COLS,
            how="inner",
            validate="one_to_one",
        )
        .merge(
            df_d[KEY_COLS + ["isMHW"]],
            on=KEY_COLS,
            how="left",
            validate="one_to_one",
        )
    )

    df["delta_gpp"] = df["gpp_cf"] - df["gpp_factual"]

    # ------------------------------------------------------------
    # Only MHW months
    # ------------------------------------------------------------
    ev = df[df["isMHW"] == 1].copy()
    if ev.empty:
        raise RuntimeError("No MHW months found (isMHW == 1).")

    # ------------------------------------------------------------
    # Direction statistics
    # ------------------------------------------------------------
    ev["direction"] = np.where(
        ev["delta_gpp"] > 0, "increase",
        np.where(ev["delta_gpp"] < 0, "decrease", "neutral")
    )

    dir_summary = (
        ev.groupby("direction")
        .agg(
            n=("delta_gpp", "count"),
            mean_delta=("delta_gpp", "mean"),
            median_delta=("delta_gpp", "median"),
        )
        .reset_index()
    )

    total_n = ev.shape[0]
    dir_summary["fraction"] = dir_summary["n"] / total_n

    # ------------------------------------------------------------
    # Distribution statistics
    # ------------------------------------------------------------
    def dist_stats(x: pd.Series):
        return {
            "n": int(x.notna().sum()),
            "mean": float(x.mean()),
            "median": float(x.median()),
            "p05": float(x.quantile(0.05)),
            "p25": float(x.quantile(0.25)),
            "p75": float(x.quantile(0.75)),
            "p95": float(x.quantile(0.95)),
        }

    dist = []

    dist.append({"metric": "delta_gpp", **dist_stats(ev["delta_gpp"])})
    dist.append({"metric": "abs_delta_gpp", **dist_stats(ev["delta_gpp"].abs())})

    dist_df = pd.DataFrame(dist)

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------
    out_dir = args.outdir

    dir_out = os.path.join(out_dir, "step5p_event_month_direction_summary.csv")
    dist_out = os.path.join(out_dir, "step5p_event_month_delta_gpp_distribution.csv")

    dir_summary.to_csv(dir_out, index=False)
    dist_df.to_csv(dist_out, index=False)

    # ------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[STEP 5p OK] Event-month GPP response (direction & distribution)")
    print("=" * 80)
    print(f"Total MHW months (grid×month): {total_n}")
    print("\n[Direction breakdown]")
    print(dir_summary.to_string(index=False))
    print("\n[Distribution summary]")
    print(dist_df.to_string(index=False))
    print("=" * 80)
    print("[OUT]")
    print(f" - {dir_out}")
    print(f" - {dist_out}")
    print("=" * 80)


if __name__ == "__main__":
    main()
