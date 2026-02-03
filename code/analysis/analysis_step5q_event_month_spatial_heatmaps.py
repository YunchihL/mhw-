#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 5q (spatial heatmaps):

Produce two heatmaps for contemporaneous ΔGPP response to MHW (isMHW=1):

Fig. A: Month × Latitude band
Fig. B: Month × Biogeographic region (IWP / East Pacific / Atlantic)

Metric:
  directional_dominance = P(ΔGPP < 0) − P(ΔGPP > 0)

ΔGPP = GPP_cf − GPP_factual
ΔGPP < 0 → positive contemporaneous response to MHW
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

KEY_COLS = ["grid_id", "year", "month"]


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def lat_band(lat, step=5):
    """Bin latitude into bands (e.g. 5-degree)."""
    return int(np.floor(lat / step) * step)


def assign_region(lon):
    """
    Very coarse biogeographic regions for mangroves.
    Longitude in degrees [-180, 180].
    """
    if 80 <= lon <= 160:
        return "IWP"
    elif -120 <= lon <= -80:
        return "East Pacific"
    else:
        return "Atlantic"


# --------------------------------------------------
# Summaries
# --------------------------------------------------
def summarize_month_lat(ev, lat_step=5, min_n=30):
    ev = ev.copy()
    ev["lat_band"] = ev["lat_c"].apply(lambda x: lat_band(x, lat_step))

    rows = []
    for (m, lb), sub in ev.groupby(["month", "lat_band"]):
        if len(sub) < min_n:
            continue
        p_neg = (sub["delta_gpp"] < 0).mean()
        p_pos = (sub["delta_gpp"] > 0).mean()
        rows.append({
            "month": m,
            "lat_band": lb,
            "directional_dominance": p_neg - p_pos,
            "n": len(sub)
        })
    return pd.DataFrame(rows)


def summarize_month_region(ev, min_n=30):
    ev = ev.copy()
    ev["region"] = ev["lon_c"].apply(assign_region)

    rows = []
    for (m, r), sub in ev.groupby(["month", "region"]):
        if len(sub) < min_n:
            continue
        p_neg = (sub["delta_gpp"] < 0).mean()
        p_pos = (sub["delta_gpp"] > 0).mean()
        rows.append({
            "month": m,
            "region": r,
            "directional_dominance": p_neg - p_pos,
            "n": len(sub)
        })
    return pd.DataFrame(rows)


# --------------------------------------------------
# Plotting
# --------------------------------------------------
def plot_heatmap(df, index, columns, title, ylabel, outpath):
    pivot = (
        df.pivot(index=index, columns=columns, values="directional_dominance")
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    norm = TwoSlopeNorm(
        vmin=pivot.min().min(),
        vcenter=0.0,
        vmax=pivot.max().max()
    )

    im = ax.imshow(
        pivot,
        cmap="RdBu_r",
        norm=norm,
        aspect="auto",
        origin="lower"
    )

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_xlabel("Month")

    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_ylabel(ylabel)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("P(ΔGPP < 0) − P(ΔGPP > 0)")

    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


# --------------------------------------------------
# Main
# --------------------------------------------------
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

    for col in ["lat_c", "lon_c", "isMHW"]:
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

    ev = df[df["isMHW"] == 1].copy()

    # ----------------------------
    # Heatmap A: Month × Latitude
    # ----------------------------
    lat_df = summarize_month_lat(ev, lat_step=5, min_n=30)
    lat_df.to_csv(
        os.path.join(args.outdir, "step5q_month_lat_table.csv"),
        index=False
    )

    plot_heatmap(
        lat_df,
        index="lat_band",
        columns="month",
        ylabel="Latitude band (°)",
        title=(
            "Seasonal dominance of contemporaneous GPP response to MHW\n"
            "across latitudinal bands"
        ),
        outpath=os.path.join(
            args.outdir,
            "step5q_heatmap_month_lat.png"
        )
    )

    # ----------------------------
    # Heatmap B: Month × Region
    # ----------------------------
    reg_df = summarize_month_region(ev, min_n=30)
    reg_df.to_csv(
        os.path.join(args.outdir, "step5q_month_region_table.csv"),
        index=False
    )

    plot_heatmap(
        reg_df,
        index="region",
        columns="month",
        ylabel="Region",
        title=(
            "Seasonal dominance of contemporaneous GPP response to MHW\n"
            "across biogeographic regions"
        ),
        outpath=os.path.join(
            args.outdir,
            "step5q_heatmap_month_region.png"
        )
    )

    print("=" * 80)
    print("[STEP 5q OK] Spatial heatmaps generated")
    print("[OUT]")
    print(" - step5q_heatmap_month_lat.png")
    print(" - step5q_heatmap_month_region.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
