# ============================================================
# A3: Inertness Check (Boxplot version)
#
# Goal (paper-friendly):
#   Compare |ΔGPP%| distributions between:
#     - non-MHW months (isMHW=0)
#     - MHW months     (isMHW=1)
#
# Why this matters:
#   If the model is reliable for counterfactual reasoning, it
#   should NOT produce spurious large changes in non-event months.
#
# Inputs:
#   1) Rolling predictions (from factual_rolling_predict.py):
#      results/predictions/factual_rolling_predictions.csv
#      must contain: grid_id, year, month, gpp_true, gpp_pred
#
#   2) Raw data.csv (for isMHW label):
#      data/data.csv (path comes from config.yaml via get_raw_data)
#      must contain: grid_id, year, month, isMHW
#
# Outputs:
#   - results/tables/A3_inertness_boxplot_summary.csv
#   - results/figures/A3_inertness_abs_delta_pct_boxplot.png
#
# Run:
#   python -m code.evaluate.a3_inertness_check_boxplot
#   python -m code.evaluate.a3_inertness_check_boxplot --pred-csv results/predictions/factual_rolling_predictions.csv
# ============================================================

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from code.train.train_tft import (
    get_config,
    get_raw_data,
)


def ensure_cols(df: pd.DataFrame, cols, name="dataframe"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in {name}: {missing}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred-csv",
        type=str,
        default="results/predictions/factual_rolling_predictions.csv",
        help="Rolling factual prediction CSV",
    )
    parser.add_argument(
        "--out-table",
        type=str,
        default="results/tables/A3_inertness_boxplot_summary.csv",
        help="Output summary table path",
    )
    parser.add_argument(
        "--out-fig",
        type=str,
        default="results/figures/A3_inertness_abs_delta_pct_boxplot.png",
        help="Output figure path",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-9,
        help="Small number to avoid division by zero",
    )
    args = parser.parse_args()

    # -----------------------------
    # 1) load rolling predictions
    # -----------------------------
    pred = pd.read_csv(args.pred_csv)
    ensure_cols(pred, ["grid_id", "year", "month", "gpp_true", "gpp_pred"], "pred-csv")

    pred["grid_id"] = pred["grid_id"].astype(int)
    pred["year"] = pred["year"].astype(int)
    pred["month"] = pred["month"].astype(int)

    # -----------------------------
    # 2) load raw data for isMHW
    # -----------------------------
    config = get_config()
    raw = get_raw_data(config)
    ensure_cols(raw, ["grid_id", "year", "month", "isMHW"], "raw data.csv")

    raw["grid_id"] = raw["grid_id"].astype(int)
    raw["year"] = raw["year"].astype(int)
    raw["month"] = raw["month"].astype(int)

    raw_key = raw[["grid_id", "year", "month", "isMHW"]].drop_duplicates(
        subset=["grid_id", "year", "month"], keep="first"
    )

    # -----------------------------
    # 3) merge
    # -----------------------------
    df = pred.merge(
        raw_key,
        on=["grid_id", "year", "month"],
        how="left",
        validate="one_to_one",
    )

    if df["isMHW"].isna().any():
        n = int(df["isMHW"].isna().sum())
        raise RuntimeError(
            f"[FATAL] {n} rows missing isMHW after merge. "
            f"Check if pred-csv and data.csv share the same grid_id/year/month."
        )

    df["isMHW"] = df["isMHW"].astype(int)

    # -----------------------------
    # 4) compute |ΔGPP%|
    # -----------------------------
    denom = np.maximum(np.abs(df["gpp_true"].values), args.eps)
    df["abs_delta_pct"] = np.abs(df["gpp_pred"].values - df["gpp_true"].values) / denom * 100.0

    non = df[df["isMHW"] == 0]["abs_delta_pct"].values
    mhw = df[df["isMHW"] == 1]["abs_delta_pct"].values

    if len(non) == 0 or len(mhw) == 0:
        raise RuntimeError("[FATAL] One of the groups is empty (nonMHW or MHW).")

    # -----------------------------
    # 5) summary stats
    # -----------------------------
    summary = pd.DataFrame(
        {
            "group": ["nonMHW", "MHW"],
            "n": [len(non), len(mhw)],
            "median_abs_delta_pct": [np.median(non), np.median(mhw)],
            "p25_abs_delta_pct": [np.percentile(non, 25), np.percentile(mhw, 25)],
            "p75_abs_delta_pct": [np.percentile(non, 75), np.percentile(mhw, 75)],
            "mean_abs_delta_pct": [np.mean(non), np.mean(mhw)],
        }
    )

    os.makedirs(os.path.dirname(args.out_table), exist_ok=True)
    summary.to_csv(args.out_table, index=False)

   # --------------------------------------------------
# 6. Histogram-based inertness plot (20% bins)
# --------------------------------------------------
bin_width = 20
max_pct = 100
bins = np.arange(0, max_pct + bin_width, bin_width)

non = df[df["isMHW"] == 0]["abs_delta_pct"].values
mhw = df[df["isMHW"] == 1]["abs_delta_pct"].values

# clip to [0, 100] for clean visualization
non = np.clip(non, 0, max_pct)
mhw = np.clip(mhw, 0, max_pct)

# normalized histograms (proportion)
non_hist, _ = np.histogram(non, bins=bins, density=True)
mhw_hist, _ = np.histogram(mhw, bins=bins, density=True)

# convert density → proportion per bin
non_prop = non_hist * bin_width
mhw_prop = mhw_hist * bin_width

bin_centers = bins[:-1] + bin_width / 2

# --------------------------------------------------
# 7. plotting
# --------------------------------------------------
os.makedirs(os.path.dirname(args.out_fig), exist_ok=True)

plt.figure(figsize=(7, 5))

plt.plot(
    bin_centers,
    non_prop,
    marker="o",
    linewidth=2,
    label="non-MHW (isMHW=0)",
)

plt.plot(
    bin_centers,
    mhw_prop,
    marker="s",
    linewidth=2,
    linestyle="--",
    label="MHW (isMHW=1)",
)

plt.xlabel(r"|ΔGPP%|")
plt.ylabel("Fraction of samples")
plt.title("A3: Inertness Check\nDistribution of |ΔGPP%| (20% bins)")
plt.xticks(bins)
plt.ylim(0, 1.0)
plt.legend(frameon=False)

plt.tight_layout()
plt.savefig(args.out_fig, dpi=300)
plt.close()


if __name__ == "__main__":
    main()
