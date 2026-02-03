#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_step1_to_step6_summary_panel.py

Create a 2x3 panel figure summarizing Step 1-6 results for presentation.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set style for better presentation
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

OUTDIR = "analysis/results"
os.makedirs(OUTDIR, exist_ok=True)

def create_summary_panel():
    """Create a 2x3 panel figure summarizing all steps."""

    # Create figure with 2x3 subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Summary of MHW Impact on Mangrove GPP (Step 1-6)',
                 fontsize=14, fontweight='bold', y=0.98)

    # ============================================================
    # Panel 1: Global annual MHW impact (Step 1)
    # ============================================================
    try:
        df1 = pd.read_csv(f"{OUTDIR}/annual_global_mhw_impact_cf_minus_factual.csv")
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in df1["impact_year"]]

        axs[0, 0].bar(df1["year"], df1["impact_year"], color=colors, width=0.6)
        axs[0, 0].axhline(0, color="k", ls="--", lw=1)
        axs[0, 0].set_xlabel("Year")
        axs[0, 0].set_ylabel("ΔGPP (cf − factual)")
        axs[0, 0].set_title("Step 1: Global annual MHW impact")
        axs[0, 0].ticklabel_format(axis='y', style='sci', scilimits=(9,9))

        # Add text annotation for direction
        increase_years = (df1["impact_year"] > 0).sum()
        total_years = len(df1)
        axs[0, 0].text(0.02, 0.98, f'{increase_years}/{total_years} years >0',
                       transform=axs[0, 0].transAxes, fontsize=8,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    except Exception as e:
        axs[0, 0].text(0.5, 0.5, f'Error loading Step 1 data:\n{str(e)}',
                       ha='center', va='center', transform=axs[0, 0].transAxes)
        axs[0, 0].set_title("Step 1: Data not available")

    # ============================================================
    # Panel 2: Spatial concentration (Step 2)
    # ============================================================
    try:
        df2 = pd.read_csv(f"{OUTDIR}/grid_mean_impact_cf_minus_factual.csv")
        df2 = df2[df2["mean_impact_year"] < 0].copy()
        df2["abs_loss"] = -df2["mean_impact_year"]
        df2 = df2.sort_values("abs_loss", ascending=False)

        top10 = df2.head(10)["abs_loss"].sum()
        total = df2["abs_loss"].sum()
        others = total - top10

        wedges, texts, autotexts = axs[0, 1].pie(
            [top10, others],
            labels=["Top 10 grids", "Other grids"],
            autopct=lambda p: f"{p:.1f}%",
            startangle=90,
            counterclock=False,
            colors=["#ff7f0e", "#c7c7c7"],
            textprops={'fontsize': 8}
        )
        axs[0, 1].set_title("Step 2: Spatial concentration of loss")

        # Add summary text
        percent_top10 = (top10 / total * 100) if total > 0 else 0
        axs[0, 1].text(0.5, -0.1, f'Top 10 grids: {percent_top10:.1f}% of total loss',
                       transform=axs[0, 1].transAxes, ha='center', fontsize=8)
    except Exception as e:
        axs[0, 1].text(0.5, 0.5, f'Error loading Step 2 data:\n{str(e)}',
                       ha='center', va='center', transform=axs[0, 1].transAxes)
        axs[0, 1].set_title("Step 2: Data not available")

    # ============================================================
    # Panel 3: Impact vs MHW duration (Step 4)
    # ============================================================
    try:
        df4 = pd.read_csv(f"{OUTDIR}/grid_year_impact_and_mhw_structure.csv")

        scatter = axs[0, 2].scatter(df4["duration_year"], df4["impact_year"],
                                   s=15, alpha=0.6, c=df4["impact_year"],
                                   cmap='RdBu_r', vmin=-1e10, vmax=1e10)
        axs[0, 2].axhline(0, color="k", ls="--", lw=1)
        axs[0, 2].set_xlabel("MHW duration (weighted sum)")
        axs[0, 2].set_ylabel("ΔGPP (annual)")
        axs[0, 2].set_title("Step 4: ΔGPP vs MHW duration")
        axs[0, 2].ticklabel_format(axis='both', style='sci', scilimits=(9,9))

        # Add correlation coefficient
        if "impact_year" in df4.columns and "duration_year" in df4.columns:
            corr = df4["impact_year"].corr(df4["duration_year"])
            axs[0, 2].text(0.02, 0.98, f'Pearson r = {corr:.3f}',
                          transform=axs[0, 2].transAxes, fontsize=8,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    except Exception as e:
        axs[0, 2].text(0.5, 0.5, f'Error loading Step 4 data:\n{str(e)}',
                       ha='center', va='center', transform=axs[0, 2].transAxes)
        axs[0, 2].set_title("Step 4: Data not available")

    # ============================================================
    # Panel 4: Distribution of ΔGPP in MHW months (Step 5p)
    # ============================================================
    try:
        # Use step5q_event_rows.csv which contains actual delta_gpp values
        df5p = pd.read_csv(f"{OUTDIR}/step5q_event_rows.csv")

        # Extract delta_gpp values
        delta_gpp_values = df5p["delta_gpp"].dropna()

        # Create histogram with more bins for better visualization
        axs[1, 0].hist(delta_gpp_values, bins=80, alpha=0.7, color='skyblue', edgecolor='black')
        axs[1, 0].axvline(0, color="r", ls="--", lw=2, label='Zero line')
        axs[1, 0].set_xlabel("ΔGPP (cf − factual)")
        axs[1, 0].set_ylabel("Count")
        axs[1, 0].set_title("Step 5p: Distribution in MHW months")
        axs[1, 0].ticklabel_format(axis='x', style='sci', scilimits=(9,9))
        axs[1, 0].legend(fontsize=8)

        # Add statistics
        mean_val = delta_gpp_values.mean()
        median_val = delta_gpp_values.median()
        neg_fraction = (delta_gpp_values < 0).mean() * 100
        axs[1, 0].text(0.02, 0.98, f'Mean: {mean_val:.2e}\nMedian: {median_val:.2e}\n{neg_fraction:.1f}% < 0',
                      transform=axs[1, 0].transAxes, fontsize=7,
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Add vertical line at mean and median
        axs[1, 0].axvline(mean_val, color='green', linestyle=':', linewidth=1.5, label=f'Mean: {mean_val:.2e}')
        axs[1, 0].axvline(median_val, color='orange', linestyle=':', linewidth=1.5, label=f'Median: {median_val:.2e}')
    except Exception as e:
        axs[1, 0].text(0.5, 0.5, f'Error loading Step 5p data:\n{str(e)}',
                       ha='center', va='center', transform=axs[1, 0].transAxes)
        axs[1, 0].set_title("Step 5p: Data not available")

    # ============================================================
    # Panel 5: Monthly pattern (Step 5q)
    # ============================================================
    try:
        df5q_m = pd.read_csv(f"{OUTDIR}/step5q_by_month.csv")

        months = df5q_m["month"]
        frac_negative = df5q_m["frac_decrease_nonzero"]  # ΔGPP < 0 means positive response

        bars = axs[1, 1].bar(months, frac_negative, color='lightcoral', edgecolor='darkred', width=0.7)
        axs[1, 1].axhline(0.5, color="k", ls="--", lw=1, alpha=0.5)
        axs[1, 1].set_xlabel("Month")
        axs[1, 1].set_ylabel("Fraction with ΔGPP < 0")
        axs[1, 1].set_title("Step 5q: Seasonal pattern")
        axs[1, 1].set_xticks(range(1, 13))
        axs[1, 1].set_ylim(0, 1)

        # Highlight months with highest positive response
        max_month = months[frac_negative.idxmax()]
        axs[1, 1].text(max_month, frac_negative.max() + 0.03, f'Max: M{max_month}',
                      ha='center', fontsize=7, fontweight='bold')
    except Exception as e:
        axs[1, 1].text(0.5, 0.5, f'Error loading Step 5q month data:\n{str(e)}',
                       ha='center', va='center', transform=axs[1, 1].transAxes)
        axs[1, 1].set_title("Step 5q: Data not available")

    # ============================================================
    # Panel 6: Latitudinal pattern (Step 5q)
    # ============================================================
    try:
        df5q_lat = pd.read_csv(f"{OUTDIR}/step5q_by_latband.csv")

        # Sort latitude bands logically
        lat_bands = df5q_lat["lat_band"].tolist()
        # Extract numeric values for sorting
        def extract_lat(s):
            try:
                return float(s.split('~')[0])
            except:
                return 0

        df5q_lat['lat_sort'] = df5q_lat['lat_band'].apply(extract_lat)
        df5q_lat = df5q_lat.sort_values('lat_sort', ascending=False)

        bands = df5q_lat["lat_band"]
        frac_negative_lat = df5q_lat["frac_decrease_nonzero"]

        bars = axs[1, 2].barh(range(len(bands)), frac_negative_lat,
                             color='lightgreen', edgecolor='darkgreen', height=0.7)
        axs[1, 2].axvline(0.5, color="k", ls="--", lw=1, alpha=0.5)
        axs[1, 2].set_yticks(range(len(bands)))
        axs[1, 2].set_yticklabels(bands)
        axs[1, 2].set_xlabel("Fraction with ΔGPP < 0")
        axs[1, 2].set_title("Step 5q: Latitudinal pattern")
        axs[1, 2].set_xlim(0, 1)

        # Highlight band with highest positive response
        max_idx = frac_negative_lat.idxmax()
        axs[1, 2].text(frac_negative_lat.max() + 0.03, max_idx, f'Max: {bands.iloc[max_idx]}',
                      va='center', fontsize=7, fontweight='bold')
    except Exception as e:
        axs[1, 2].text(0.5, 0.5, f'Error loading Step 5q lat data:\n{str(e)}',
                       ha='center', va='center', transform=axs[1, 2].transAxes)
        axs[1, 2].set_title("Step 5q: Data not available")

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle

    # Save figure
    output_path = f"{OUTDIR}/fig_step1_to_step6_summary_panel.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[SUCCESS] Summary panel saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    print("==============================================")
    print("Generating Step 1-6 summary panel figure")
    print("==============================================")

    output_path = create_summary_panel()

    print("==============================================")
    print(f"Figure saved: {output_path}")
    print("==============================================")