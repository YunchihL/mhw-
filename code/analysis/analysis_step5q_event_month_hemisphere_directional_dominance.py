#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_step5q_event_month_hemisphere_directional_dominance.py

Step 5q (final + spatial context):
MHW 当月（isMHW=1）ΔGPP 的时间分布，
分别在北半球（NH）和南半球（SH）中分析，
并在每个面板中展示 grid 的纬度与经度分布（insets）。

Metric:
  directional_dominance
  = P(ΔGPP < 0) − P(ΔGPP > 0)

Definition:
  ΔGPP = GPP_cf − GPP_factual
  ΔGPP < 0 → positive contemporaneous response to MHW
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

KEY_COLS = ["grid_id", "year", "month"]


def assign_hemisphere(lat):
    if lat > 0:
        return "NH"
    elif lat < 0:
        return "SH"
    else:
        return "EQ"


def summarize_directional_dominance(df):
    rows = []
    for m, sub in df.groupby("month"):
        n = len(sub)
        if n == 0:
            continue
        p_neg = (sub["delta_gpp"] < 0).mean()
        p_pos = (sub["delta_gpp"] > 0).mean()
        rows.append({
            "month": m,
            "n": n,
            "p_delta_neg": p_neg,
            "p_delta_pos": p_pos,
            "directional_dominance": p_neg - p_pos
        })
    return pd.DataFrame(rows).sort_values("month")


def plot_two_panel_line_with_latlon(nh, sh, ev, outpath):
    fig, axes = plt.subplots(
        nrows=2, ncols=1,
        figsize=(10, 7),
        sharex=True,
        constrained_layout=True
    )

    for ax, df, hemi, title, color in zip(
        axes,
        [nh, sh],
        ["NH", "SH"],
        ["Northern Hemisphere", "Southern Hemisphere"],
        ["#1f4e79", "#8b0000"]  # NH blue, SH dark red
    ):
        # ----------------------------
        # Main line + point plot
        # ----------------------------
        sizes = 40 + 160 * (df["n"] / df["n"].max())

        ax.plot(
            df["month"],
            df["directional_dominance"],
            color=color,
            linewidth=2
        )
        ax.scatter(
            df["month"],
            df["directional_dominance"],
            s=sizes,
            color=color,
            edgecolor="black",
            zorder=3
        )

        ax.axhline(0, linestyle="--", color="black", linewidth=1)
        ax.set_ylabel("P(ΔGPP < 0) − P(ΔGPP > 0)")
        ax.set_title(title)

        # ----------------------------
        # Inset 1: latitude distribution
        # ----------------------------
        sub = ev[ev["hemisphere"] == hemi]

        ax_lat = inset_axes(
            ax,
            width="30%", height="30%",
            loc="upper right",
            borderpad=1.1
        )
        ax_lat.hist(
            sub["lat_c"],
            bins=15,
            orientation="horizontal",
            color=color,
            alpha=0.75
        )
        ax_lat.axhline(0, linestyle="--", color="gray", linewidth=0.8)
        ax_lat.set_xticks([])
        ax_lat.set_yticks([])
        ax_lat.set_title("Lat", fontsize=8)

        # ----------------------------
        # Inset 2: longitude distribution
        # ----------------------------
        ax_lon = inset_axes(
            ax,
            width="30%", height="30%",
            loc="lower right",
            borderpad=1.1
        )
        ax_lon.hist(
            sub["lon_c"],
            bins=20,
            orientation="horizontal",
            color=color,
            alpha=0.75
        )
        ax_lon.set_xticks([])
        ax_lon.set_yticks([])
        ax_lon.set_title("Lon", fontsize=8)

    axes[-1].set_xlabel("Month")

    fig.suptitle(
        "Seasonal dominance of contemporaneous GPP response to MHW\n"
        "with latitudinal and longitudinal distribution of grids (insets)",
        fontsize=12
    )

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

    # ----------------------------
    # Load
    # ----------------------------
    df_f = pd.read_csv(args.factual)
    df_c = pd.read_csv(args.counterfactual)
    df_d = pd.read_csv(args.data)

    for col in ["lat_c", "lon_c"]:
        if col not in df_d.columns:
            raise ValueError(f"data.csv 必须包含 {col}")

    # ----------------------------
    # Merge
    # ----------------------------
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
            df_d[KEY_COLS + ["isMHW", "lat_c", "lon_c"]],
            on=KEY_COLS,
            how="left",
            validate="one_to_one",
        )
    )

    df["delta_gpp"] = df["gpp_cf"] - df["gpp_factual"]
    df["hemisphere"] = df["lat_c"].apply(assign_hemisphere)

    ev = df[
        (df["isMHW"] == 1) &
        (df["hemisphere"].isin(["NH", "SH"]))
    ].copy()

    # ----------------------------
    # Summaries
    # ----------------------------
    nh = summarize_directional_dominance(ev[ev["hemisphere"] == "NH"])
    sh = summarize_directional_dominance(ev[ev["hemisphere"] == "SH"])

    nh.to_csv(os.path.join(args.outdir, "step5q_by_month_NH.csv"), index=False)
    sh.to_csv(os.path.join(args.outdir, "step5q_by_month_SH.csv"), index=False)

    # ----------------------------
    # Plot
    # ----------------------------
    figpath = os.path.join(
        args.outdir,
        "step5q_monthly_directional_dominance_NH_SH_line_with_latlon.png"
    )
    plot_two_panel_line_with_latlon(nh, sh, ev, figpath)

    print("=" * 80)
    print("[STEP 5q FINAL OK] Hemisphere-aware seasonal dominance")
    print("with latitudinal and longitudinal grid distributions")
    print("[OUT]")
    print(" - step5q_by_month_NH.csv")
    print(" - step5q_by_month_SH.csv")
    print(" - step5q_monthly_directional_dominance_NH_SH_line_with_latlon.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
