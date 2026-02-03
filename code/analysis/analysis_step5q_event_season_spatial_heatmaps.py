#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 5q (seasonal spatial heatmaps)

Produce two SEASONAL heatmaps for contemporaneous ΔGPP response to MHW (isMHW=1):

Fig. 4.4a: Season × Latitude band
Fig. Sx:   Season × Biogeographic region (IWP / East Pacific / Atlantic)

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

SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def month_to_season(month):
    """Convert month (1–12) to climatological season."""
    if month in [12, 1, 2]:
        return "DJF"
    elif month in [3, 4, 5]:
        return "MAM"
    elif month in [6, 7, 8]:
        return "JJA"
    elif month in [9, 10, 11]:
        return "SON"
    else:
        raise ValueError("Invalid month")


def lat_band(lat, step=10):
    """Bin latitude into coarse bands (default 10°)."""
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
def summarize_season_lat(ev, lat_step=10, min_n=50):
    ev = ev.copy()
    ev["season"] = ev["month"].apply(month_to_season)
    ev["lat_band"] = ev["lat_c"].apply(lambda x: lat_band(x, lat_step))

    rows = []
    for (s, lb), sub in ev.groupby(["season", "lat_band"]):
        if len(sub) < min_n:
            continue
        p_neg = (sub["delta_gpp"] < 0).mean()
        p_pos = (sub["delta_gpp"] > 0).mean()
        rows.append({
            "season": s,
            "lat_band": lb,
            "directional_dominance": p_neg - p_pos,
            "n": len(sub)
        })
    return pd.DataFrame(rows)


def summarize_season_region(ev, min_n=50):
    ev = ev.copy()
    ev["season"] = ev["month"].apply(month_to_season)
    ev["region"] = ev["lon_c"].apply(assign_region)

    rows = []
    for (s, r), sub in ev.groupby(["season", "region"]):
        if len(sub) < min_n:
            continue
        p_neg = (sub["delta_gpp"] < 0).mean()
        p_pos = (sub["delta_gpp"] > 0).mean()
        rows.append({
            "season": s,
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
        .reindex(columns=SEASON_ORDER)
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5)

    im = ax.imshow(
        pivot,
        cmap="RdBu_r",
        norm=norm,
        aspect="auto",
        origin="lower"
    )

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_xlabel("Season")

    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_ylabel(ylabel)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
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
            raise ValueError(f"data.csv must contain {col}")

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
    # Seasonal heatmap: latitude
    # ----------------------------
    lat_df = summarize_season_lat(ev, lat_step=10, min_n=50)
    lat_df.to_csv(
        os.path.join(args.outdir, "step5q_season_lat_table.csv"),
        index=False
    )

    plot_heatmap(
        lat_df,
        index="lat_band",
        columns="season",
        ylabel="Latitude band (°)",
        title=(
            "Seasonal dominance of contemporaneous GPP response to MHW\n"
            "across latitudinal bands"
        ),
        outpath=os.path.join(
            args.outdir,
            "step5q_heatmap_season_lat.png"
        )
    )

    # ----------------------------
    # Seasonal heatmap: region
    # ----------------------------
    reg_df = summarize_season_region(ev, min_n=50)
    reg_df.to_csv(
        os.path.join(args.outdir, "step5q_season_region_table.csv"),
        index=False
    )

    plot_heatmap(
        reg_df,
        index="region",
        columns="season",
        ylabel="Region",
        title=(
            "Seasonal dominance of contemporaneous GPP response to MHW\n"
            "across biogeographic regions"
        ),
        outpath=os.path.join(
            args.outdir,
            "step5q_heatmap_season_region.png"
        )
    )

    print("=" * 80)
    print("[STEP 5q OK] Seasonal spatial heatmaps generated")
    print("[OUT]")
    print(" - step5q_heatmap_season_lat.png")
    print(" - step5q_heatmap_season_region.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
