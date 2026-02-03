# ============================================================
# Step 7B: MHW structure × GPP response (2×2 binning)
# Pairwise combinations among:
#   - duration_weighted_sum
#   - intensity_cumulative_weighted_sum
#   - intensity_density
# Event months only (isMHW = 1)
# Median-based binning (50/50) for robustness
# ============================================================
# python code/analysis/analysis_step7b_mhw_structure_binning.py \
#   --factual results/predictions/factual_rolling_predictions.csv \
#   --counterfactual results/predictions/counterfactual_rolling_predictions.csv \
#   --data data/data.csv \
#   --outdir code/analysis/results/step7b_structure


import argparse
import os
import pandas as pd
import numpy as np


STRUCT_VARS = {
    "duration_weighted_sum": "Duration",
    "intensity_cumulative_weighted_sum": "Intensity",
    "intensity_density": "IntensityDensity",
}


def load_and_merge(factual, counterfactual, data):
    df_f = pd.read_csv(factual)
    df_cf = pd.read_csv(counterfactual)
    df_d = pd.read_csv(data)

    keys = ["grid_id", "year", "month"]
    df = df_f.merge(
        df_cf[keys + ["gpp_pred"]],
        on=keys,
        suffixes=("_factual", "_cf"),
    )
    df = df.merge(
        df_d[keys + list(STRUCT_VARS.keys()) + ["isMHW"]],
        on=keys,
        how="left",
    )

    # loss definition aligned with hazard view
    df["loss_abs"] = df["gpp_pred_factual"] - df["gpp_pred_cf"]

    eps = np.percentile(np.abs(df["gpp_pred_factual"]), 5)
    df["loss_pct"] = df["loss_abs"] / (np.abs(df["gpp_pred_factual"]) + eps) * 100

    return df


def two_by_two_binning(df, x, y):
    df = df.copy()

    x_med = df[x].median()
    y_med = df[y].median()

    df["x_bin"] = np.where(df[x] >= x_med, "high", "low")
    df["y_bin"] = np.where(df[y] >= y_med, "high", "low")

    records = []

    for xb in ["low", "high"]:
        for yb in ["low", "high"]:
            sub = df[(df["x_bin"] == xb) & (df["y_bin"] == yb)]

            if len(sub) == 0:
                continue

            records.append({
                "x": x,
                "y": y,
                "x_level": xb,
                "y_level": yb,
                "n": len(sub),
                "loss_pct_median": sub["loss_pct"].median(),
                "loss_pct_p25": sub["loss_pct"].quantile(0.25),
                "loss_pct_p75": sub["loss_pct"].quantile(0.75),
                "damage_ratio": (sub["loss_abs"] > 0).mean(),
            })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factual", required=True)
    parser.add_argument("--counterfactual", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_and_merge(args.factual, args.counterfactual, args.data)

    # event months only
    df_evt = df[df["isMHW"] == 1].reset_index(drop=True)

    results = []

    vars_list = list(STRUCT_VARS.keys())
    for i in range(len(vars_list)):
        for j in range(i + 1, len(vars_list)):
            x = vars_list[i]
            y = vars_list[j]
            res = two_by_two_binning(df_evt, x, y)
            results.append(res)

    out = pd.concat(results, ignore_index=True)

    out.to_csv(
        os.path.join(args.outdir, "step7b_mhw_structure_2x2_bins.csv"),
        index=False,
    )

    print("============================================================")
    print("[STEP 7B] MHW structure × GPP response (2×2 binning)")
    print("Binning: median (50/50)")
    print("Event months only (isMHW = 1)")
    print("------------------------------------------------------------")
    print(out)
    print("============================================================")
    print(f"[OUT] {args.outdir}/step7b_mhw_structure_2x2_bins.csv")


if __name__ == "__main__":
    main()
