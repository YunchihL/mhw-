#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_step5q_event_month_hemisphere_barplot.py

Step 5q (revised):
MHW 当月（isMHW=1）ΔGPP 的时间分布，
分别在北半球（NH）和南半球（SH）中分析。

Definition:
  ΔGPP = GPP_cf − GPP_factual
  ΔGPP < 0  → positive contemporaneous response to MHW
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

KEY_COLS = ["grid_id", "year", "month"]


def assign_hemisphere(lat):
    if lat > 0:
        return "NH"
    elif lat < 0:
        return "SH"
    else:
        return "EQ"


def summarize_by_month(df):
    rows = []
    for m, sub in df.groupby("month"):
        n = len(sub)
        n_pos = (sub["delta_gpp"] < 0).sum()   # 正向：ΔGPP < 0
        rows.append({
            "month": m,
            "n": n,
            "n_delta_neg": int(n_pos),
            "frac_delta_neg": n_pos / n if n > 0 else np.nan
        })
    return pd.DataFrame(rows).sort_values("month")


def plot_bar(df_nh, df_sh, outpath):
    months = np.arange(1, 13)

    y_nh = df_nh.set_index("month").reindex(months)["frac_delta_neg"]
    y_sh = df_sh.set_index("month").reindex(months)["frac_delta_neg"]

    width = 0.38

    plt.figure(figsize=(10, 5))
    plt.bar(months - width/2, y_nh, width=width, label="Northern Hemisphere")
    plt.bar(months + width/2, y_sh, width=width, label="Southern Hemisphere")

    plt.axhline(0.5, linestyle="--", linewidth=1, color="gray")

    plt.xticks(months)
    plt.ylim(0, 1)
    plt.xlabel("Month")
    plt.ylabel("Fraction of MHW months with ΔGPP < 0")
    plt.title("Seasonal pattern of positive contemporaneous GPP response to MHW")

    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--factual", required=True)
    p.add_argument("--counterfactual", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    df_f = pd.read_csv(args.factual)
    df_c = pd.read_csv(args.counterfactual)
    df_d = pd.read_csv(args.data)

    if "lat_c" not in df_d.columns:
        raise ValueError("data.csv 必须包含 lat_c 列")

    # --------------------------------------------------
    # Merge
    # --------------------------------------------------
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
            df_d[KEY_COLS + ["isMHW", "lat_c"]],
            on=KEY_COLS,
            how="left",
            validate="one_to_one",
        )
    )

    # ΔGPP
    df["delta_gpp"] = df["gpp_cf"] - df["gpp_factual"]

    # Hemisphere
    df["hemisphere"] = df["lat_c"].apply(assign_hemisphere)

    # Only MHW months, exclude equator
    ev = df[(df["isMHW"] == 1) & (df["hemisphere"].isin(["NH", "SH"]))].copy()

    # --------------------------------------------------
    # Summaries
    # --------------------------------------------------
    nh = summarize_by_month(ev[ev["hemisphere"] == "NH"])
    sh = summarize_by_month(ev[ev["hemisphere"] == "SH"])

    nh.to_csv(os.path.join(args.outdir, "step5q_by_month_NH.csv"), index=False)
    sh.to_csv(os.path.join(args.outdir, "step5q_by_month_SH.csv"), index=False)

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    figpath = os.path.join(
        args.outdir,
        "step5q_monthly_fraction_delta_neg_NH_SH.png"
    )
    plot_bar(nh, sh, figpath)

    print("=" * 80)
    print("[STEP 5q REVISED OK] Hemisphere-aware monthly analysis")
    print("Definition: ΔGPP < 0 → positive contemporaneous response to MHW")
    print("[OUT]")
    print(" - step5q_by_month_NH.csv")
    print(" - step5q_by_month_SH.csv")
    print(" - step5q_monthly_fraction_delta_neg_NH_SH.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
