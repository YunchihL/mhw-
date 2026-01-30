# ============================================================
# A4: Dose–Response Check (MHW intensity vs GPP response)
#
# Purpose:
#   - Test whether predicted GPP response increases with
#     Marine Heatwave (MHW) intensity
#
# Key definition:
#   intensity_density =
#       intensity_cumulative_weighted_sum
#       ---------------------------------
#            duration_weighted_sum
#
# Logic:
#   1) Merge rolling factual predictions with data.csv
#   2) Keep ONLY MHW months (isMHW == 1)
#   3) Use intensity_density from data.csv
#   4) Define response as |ΔGPP| (%)
#   5) Plot intensity_density vs |ΔGPP| (%)
#
# Input:
#   - results/predictions/factual_rolling_predictions.csv
#   - data/data.csv
#
# Output:
#   - results/figures/A4_dose_response_scatter.png
#   - results/tables/A4_dose_response_summary.csv
# ============================================================

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred-csv",
        type=str,
        default="results/predictions/factual_rolling_predictions.csv",
        help="Rolling factual prediction results",
    )
    parser.add_argument(
        "--data-csv",
        type=str,
        default="data/data.csv",
        help="Original data.csv with MHW metrics",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="Base output directory",
    )
    parser.add_argument(
        "--frac",
        type=float,
        default=0.3,
        help="LOWESS smoothing fraction",
    )
    parser.add_argument(
        "--signed",
        action="store_true",
        help="Analyze signed ΔGPP (positive/negative) instead of absolute values",
    )
    args = parser.parse_args()

    # --------------------------------------------------
    # 1. load data
    # --------------------------------------------------
    df_pred = pd.read_csv(args.pred_csv)
    df_raw = pd.read_csv(args.data_csv)

    required_pred = [
        "grid_id", "year", "month",
        "gpp_true", "gpp_pred"
    ]
    required_raw = [
        "grid_id", "year", "month",
        "isMHW",
        "intensity_density",
    ]

    for cols, name, df in [
        (required_pred, "prediction", df_pred),
        (required_raw, "raw", df_raw),
    ]:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"Missing columns in {name} data: {missing}")

    # --------------------------------------------------
    # 2. merge & keep MHW months
    # --------------------------------------------------
    df = df_pred.merge(
        df_raw[required_raw],
        on=["grid_id", "year", "month"],
        how="left"
    )

    df = df[df["isMHW"] == 1].copy()

    # Additional filter: remove MHW samples with intensity_density == 0
    # These are data inconsistencies where isMHW=1 but no actual MHW intensity
    initial_mhw_count = len(df)
    df = df[df["intensity_density"] > 0].copy()
    filtered_count = initial_mhw_count - len(df)

    if filtered_count > 0:
        print(f"[INFO] Filtered out {filtered_count} MHW samples with intensity_density = 0 (data inconsistency)")
        print(f"[INFO] Valid MHW samples: {len(df)} (originally {initial_mhw_count})")

    if len(df) == 0:
        raise RuntimeError("No MHW samples found after filtering")

    # --------------------------------------------------
    # 4. define response variable
    # --------------------------------------------------
    df["delta_gpp_pct"] = (
        (df["gpp_pred"] - df["gpp_true"]) / df["gpp_true"] * 100
    )
    df["abs_delta_gpp_pct"] = df["delta_gpp_pct"].abs()

    # clean invalid
    df = df.replace([np.inf, -np.inf], np.nan)
    # Always clean based on intensity_density and delta_gpp_pct
    df = df.dropna(subset=["intensity_density", "delta_gpp_pct"])

    # --------------------------------------------------
    # 5. LOWESS smoothing (both signed and absolute)
    # --------------------------------------------------
    # For absolute values
    lowess_curve_abs = lowess(
        df["abs_delta_gpp_pct"],
        df["intensity_density"],
        frac=args.frac,
        return_sorted=True
    )

    # For signed values
    lowess_curve_signed = lowess(
        df["delta_gpp_pct"],
        df["intensity_density"],
        frac=args.frac,
        return_sorted=True
    )


    # --------------------------------------------------
    # 6. create output directories
    # --------------------------------------------------
    fig_dir = os.path.join(args.out_dir, "figures")
    tab_dir = os.path.join(args.out_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    # --------------------------------------------------
    # 7. generate absolute value plot
    # --------------------------------------------------
    plt.figure(figsize=(7, 5))

    # scatter: individual MHW months
    plt.scatter(
        df["intensity_density"],
        df["abs_delta_gpp_pct"],
        s=10,
        alpha=0.3,
        label="MHW months"
    )

    # LOWESS trend
    plt.plot(
        lowess_curve_abs[:, 0],
        lowess_curve_abs[:, 1],
        color="red",
        linewidth=2,
        label="LOWESS trend"
    )

    plt.xlabel("MHW intensity density")
    plt.ylabel("|ΔGPP| (%)")

    # Fixed y-axis range
    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 101, 20))

    plt.title("A4: Dose–Response Check - Absolute ΔGPP (MHW months only)")
    plt.legend(frameon=False)
    plt.tight_layout()

    fig_path_abs = os.path.join(fig_dir, "A4_dose_response_scatter_abs.png")
    plt.savefig(fig_path_abs, dpi=300)
    plt.close()

    # --------------------------------------------------
    # 8. generate signed value plot
    # --------------------------------------------------
    plt.figure(figsize=(7, 5))

    # Create a colormap based on sign of delta_gpp_pct
    colors = np.where(df["delta_gpp_pct"] >= 0, "blue", "red")
    alphas = np.where(df["delta_gpp_pct"] >= 0, 0.3, 0.3)

    # scatter with color coding
    plt.scatter(
        df["intensity_density"],
        df["delta_gpp_pct"],
        s=10,
        c=colors,
        alpha=alphas,
        label="MHW months"
    )

    # LOWESS trend for signed values
    plt.plot(
        lowess_curve_signed[:, 0],
        lowess_curve_signed[:, 1],
        color="green",
        linewidth=2,
        label="LOWESS trend"
    )

    # Add horizontal line at y=0
    plt.axhline(y=0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)

    plt.xlabel("MHW intensity density")
    plt.ylabel("ΔGPP (%)")

    # Set symmetric y-axis limits based on percentiles
    y_abs_max = np.percentile(np.abs(df["delta_gpp_pct"]), 95)
    plt.ylim(-y_abs_max, y_abs_max)

    plt.title("A4: Dose–Response Check - Signed ΔGPP (MHW months only)")
    plt.legend(frameon=False)
    plt.tight_layout()

    fig_path_signed = os.path.join(fig_dir, "A4_dose_response_scatter_signed.png")
    plt.savefig(fig_path_signed, dpi=300)
    plt.close()

    # --------------------------------------------------
    # 9. summary tables
    # --------------------------------------------------
    # Absolute value summary
    summary_abs = (
        df["abs_delta_gpp_pct"]
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        .to_frame(name="|ΔGPP| (%)")
    )
    tab_path_abs = os.path.join(tab_dir, "A4_dose_response_summary_abs.csv")
    summary_abs.to_csv(tab_path_abs)

    # Signed value summary
    summary_signed = (
        df["delta_gpp_pct"]
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        .to_frame(name="ΔGPP (%)")
    )
    tab_path_signed = os.path.join(tab_dir, "A4_dose_response_summary_signed.csv")
    summary_signed.to_csv(tab_path_signed)

    # Additional signed statistics
    pos_count = (df["delta_gpp_pct"] > 0).sum()
    neg_count = (df["delta_gpp_pct"] < 0).sum()
    zero_count = (df["delta_gpp_pct"] == 0).sum()
    total_count = len(df)

    # --------------------------------------------------
    # 10. log output
    # --------------------------------------------------
    print("\n========== A4: Dose–Response Check ==========")
    print(f"N (MHW months): {total_count}")

    if args.signed:
        print(f"\n--- Signed ΔGPP Analysis ---")
        print(f"Median ΔGPP (%): {summary_signed.loc['50%', 'ΔGPP (%)']:.2f}")
        print(f"Mean ΔGPP (%): {summary_signed.loc['mean', 'ΔGPP (%)']:.2f}")
        print(f"Positive ΔGPP: {pos_count} ({pos_count/total_count*100:.1f}%)")
        print(f"Negative ΔGPP: {neg_count} ({neg_count/total_count*100:.1f}%)")
        print(f"Zero ΔGPP: {zero_count} ({zero_count/total_count*100:.1f}%)")
    else:
        print(f"Median |ΔGPP| (%): {summary_abs.loc['50%', '|ΔGPP| (%)']:.2f}")

    print("\n--- Absolute ΔGPP ---")
    print(f"Median |ΔGPP| (%): {summary_abs.loc['50%', '|ΔGPP| (%)']:.2f}")
    print("=============================================")

    print(f"\n[INFO] Saved figures:")
    print(f"  - Absolute: {fig_path_abs}")
    print(f"  - Signed: {fig_path_signed}")
    print(f"\n[INFO] Saved tables:")
    print(f"  - Absolute: {tab_path_abs}")
    print(f"  - Signed: {tab_path_signed}")


if __name__ == "__main__":
    main()
