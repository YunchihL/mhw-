#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_step1_to_step6_summary_figures.py

One-click plotting script for Step 1–6 results
Used for advisor presentation & sanity check
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUTDIR = "analysis/results"
os.makedirs(OUTDIR, exist_ok=True)

# ============================================================
# Step 1 | Global annual MHW impact (bar chart)
# ============================================================
def plot_step1():
    df = pd.read_csv(
        f"{OUTDIR}/annual_global_mhw_impact_cf_minus_factual.csv"
    )

    colors = ["#d62728" if v > 0 else "#1f77b4" for v in df["impact_year"]]

    plt.figure(figsize=(8, 4))
    plt.bar(df["year"], df["impact_year"], color=colors)
    plt.axhline(0, color="k", ls="--", lw=1)
    plt.xlabel("Year")
    plt.ylabel("ΔGPP (cf − factual)")
    plt.title("Global annual MHW-associated GPP impact")

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig_step1_global_annual_impact_bar.png", dpi=200)
    plt.close()
    print("[OK] Step 1")


# ============================================================
# Step 2 | Spatial concentration (pie chart)
# ============================================================
def plot_step2():
    df = pd.read_csv(
        f"{OUTDIR}/grid_mean_impact_cf_minus_factual.csv"
    )

    df = df[df["mean_impact_year"] < 0].copy()
    df["abs_loss"] = -df["mean_impact_year"]
    df = df.sort_values("abs_loss", ascending=False)

    top10 = df.head(10)["abs_loss"].sum()
    total = df["abs_loss"].sum()
    others = total - top10

    plt.figure(figsize=(5, 5))
    plt.pie(
        [top10, others],
        labels=["Top 10 grids", "Other grids"],
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        counterclock=False,
        colors=["#ff7f0e", "#c7c7c7"],
    )
    plt.title("Contribution of top grids to total MHW-associated GPP loss")

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig_step2_spatial_concentration_pie.png", dpi=200)
    plt.close()
    print("[OK] Step 2")


# ============================================================
# Step 4 | Impact vs MHW structure
# ============================================================
def plot_step4():
    df = pd.read_csv(
        f"{OUTDIR}/grid_year_impact_and_mhw_structure.csv"
    )

    pairs = [
        ("duration_year", "Duration (weighted sum)"),
        ("intensity_year", "Cumulative intensity"),
        ("density_year", "Intensity density"),
    ]

    for col, label in pairs:
        plt.figure(figsize=(5, 4))
        plt.scatter(df[col], df["impact_year"], s=10, alpha=0.4)
        plt.axhline(0, color="k", ls="--", lw=1)
        plt.xlabel(label)
        plt.ylabel("ΔGPP (annual)")
        plt.title(f"ΔGPP vs {label}")
        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/fig_step4_impact_vs_{col}.png", dpi=200)
        plt.close()

    print("[OK] Step 4")


# ============================================================
# Step 5p | Distribution of ΔGPP in MHW months
# ============================================================
def plot_step5p():
    # Use step5q_event_rows.csv which contains actual delta_gpp values
    df = pd.read_csv(f"{OUTDIR}/step5q_event_rows.csv")

    plt.figure(figsize=(7, 4))
    plt.hist(df["delta_gpp"], bins=80, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(0, color="r", ls="--", lw=2, label='Zero line')
    plt.xlabel("ΔGPP (cf − factual)")
    plt.ylabel("Count")
    plt.title("Distribution of ΔGPP in MHW months")
    plt.ticklabel_format(axis='x', style='sci', scilimits=(9,9))
    plt.legend()

    # Add statistics text
    mean_val = df["delta_gpp"].mean()
    median_val = df["delta_gpp"].median()
    neg_fraction = (df["delta_gpp"] < 0).mean() * 100
    plt.text(0.02, 0.98, f'Mean: {mean_val:.2e}\nMedian: {median_val:.2e}\n{neg_fraction:.1f}% < 0',
             transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig_step5p_delta_distribution.png", dpi=200)
    plt.close()
    print("[OK] Step 5p")


# ============================================================
# Step 5q | Monthly & latitudinal patterns
# ============================================================
def plot_step5q():
    # Month
    df_m = pd.read_csv(f"{OUTDIR}/step5q_by_month.csv")

    plt.figure(figsize=(7, 4))
    plt.bar(df_m["month"], df_m["frac_decrease_nonzero"])
    plt.xlabel("Month")
    plt.ylabel("Fraction of ΔGPP < 0")
    plt.title("Seasonal dependence of positive MHW response")

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig_step5q_monthly_direction.png", dpi=200)
    plt.close()

    # Latitude
    df_lat = pd.read_csv(f"{OUTDIR}/step5q_by_latband.csv")

    plt.figure(figsize=(7, 4))
    plt.bar(df_lat["lat_band"], df_lat["frac_decrease_nonzero"])
    plt.xticks(rotation=45)
    plt.xlabel("Latitude band")
    plt.ylabel("Fraction of ΔGPP < 0")
    plt.title("Latitudinal pattern of positive MHW response")

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig_step5q_latband_direction.png", dpi=200)
    plt.close()

    print("[OK] Step 5q")


# ============================================================
# Step 6 | Episode-level MHW response
# ============================================================
def plot_step6():
    # Load episode phase summary data
    step6_path = "code/analysis/results/step6p_episode_response/step6p_episode_phase_summary.csv"
    df = pd.read_csv(step6_path)

    # Define phase order for plotting
    phase_order = ['during', 'post_1', 'post_2', 'post_3', 'post_4', 'post_5', 'post_6', 'post_mean_1_3']
    # Filter and sort
    df = df[df['phase'].isin(phase_order)].copy()
    df['phase'] = pd.Categorical(df['phase'], categories=phase_order, ordered=True)
    df = df.sort_values('phase')

    plt.figure(figsize=(8, 4))
    bars = plt.bar(df['phase'], df['mean'], color='plum', edgecolor='darkviolet', width=0.7)
    plt.axhline(0, color="k", ls="--", lw=1)
    plt.xlabel("Phase relative to MHW")
    plt.ylabel("Mean ΔGPP (cf − factual)")
    plt.title("Episode-level MHW response")
    plt.ticklabel_format(axis='y', style='sci', scilimits=(9,9))
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar, mean_val in zip(bars, df['mean']):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        y_offset = 0.01 * (plt.ylim()[1] - plt.ylim()[0])
        if height >= 0:
            label_y = height + y_offset
        else:
            label_y = height - y_offset
        plt.text(bar.get_x() + bar.get_width()/2, label_y,
                f'{mean_val:.2e}', ha='center', va=va, fontsize=8)

    # Add statistics annotation
    negative_phases = (df['mean'] < 0).sum()
    total_phases = len(df)
    plt.text(0.02, 0.98, f'{negative_phases}/{total_phases} phases <0',
            transform=plt.gca().transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig_step6_episode_response.png", dpi=200)
    plt.close()
    print("[OK] Step 6")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("==============================================")
    print("Generating Step 1–6 summary figures")
    print("==============================================")

    plot_step1()
    plot_step2()
    plot_step4()
    plot_step5p()
    plot_step5q()
    plot_step6()

    print("==============================================")
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print("==============================================")
