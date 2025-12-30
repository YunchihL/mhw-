# ============================================================
# build_counterfactual_data.py
#
# Purpose
# -------
# Build a counterfactual dataset for TFT rolling prediction:
#   - Same structure as data.csv
#   - Same (grid_id, year, month, time_idx)
#   - SST replaced by unreal_sst ONLY during MHW months
#   - All MHW-related variables set to zero
#
# Output
# ------
#   data/data_counterfactual.csv
#
# This file can be directly used by:
#   factual_rolling_predict.py
#
# Usage
# -----
# python -m code.evaluate.build_counterfactual_data \
#   --data data/data.csv \
#   --unreal-sst data/grid_monthly_unreal_sst_final.csv \
#   --out data/data_counterfactual.csv
# ============================================================

import os
import argparse
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Fatal error helper
# ------------------------------------------------------------
def FATAL(msg: str):
    raise RuntimeError(f"\n[FATAL] {msg}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="data/data.csv",
        help="Original factual data.csv",
    )
    parser.add_argument(
        "--unreal-sst",
        type=str,
        required=True,
        help="Unreal SST CSV: grid_id, year, month, unreal_sst",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/data_counterfactual.csv",
        help="Output counterfactual data.csv",
    )
    args = parser.parse_args()

    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------
    if not os.path.exists(args.data):
        FATAL(f"data file not found: {args.data}")
    if not os.path.exists(args.unreal_sst):
        FATAL(f"unreal_sst file not found: {args.unreal_sst}")

    df = pd.read_csv(args.data)
    unreal = pd.read_csv(args.unreal_sst)

    print(f"[INFO] Loaded data shape: {df.shape}")

    # --------------------------------------------------
    # 2. Check required columns
    # --------------------------------------------------
    need_data = {
        "grid_id", "year", "month",
        "mean_sst", "isMHW",
        "intensity_max_month",
        "duration_weighted_sum",
        "intensity_cumulative_weighted_sum",
    }
    miss = need_data - set(df.columns)
    if miss:
        FATAL(f"data.csv missing columns: {sorted(miss)}")

    need_unreal = {"grid_id", "year", "month", "unreal_sst"}
    miss = need_unreal - set(unreal.columns)
    if miss:
        FATAL(f"unreal_sst.csv missing columns: {sorted(miss)}")

    # --------------------------------------------------
    # 3. Normalize dtypes for safe merge
    # --------------------------------------------------
    for c in ["grid_id", "year", "month"]:
        df[c] = df[c].astype(int)
        unreal[c] = unreal[c].astype(int)

    # --------------------------------------------------
    # 4. Merge unreal_sst
    # --------------------------------------------------
    df = df.merge(
        unreal[["grid_id", "year", "month", "unreal_sst"]],
        on=["grid_id", "year", "month"],
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------
    # 5. Replace SST ONLY during MHW months
    # --------------------------------------------------
    mask = (df["isMHW"] == 1) & (~df["unreal_sst"].isna())
    print(f"[INFO] Replacing mean_sst for {mask.sum()} MHW months")

    df.loc[mask, "mean_sst"] = df.loc[mask, "unreal_sst"]

    # --------------------------------------------------
    # 6. Zero out all MHW-related variables
    # --------------------------------------------------
    mhw_vars = [
        "isMHW",
        "intensity_max_month",
        "duration_weighted_sum",
        "intensity_cumulative_weighted_sum",
    ]

    print(f"[INFO] Zeroing MHW-related variables: {mhw_vars}")
    for v in mhw_vars:
        df[v] = 0

    if "intensity_density" in df.columns:
        df["intensity_density"] = np.where(
            df["duration_weighted_sum"] > 0,
            df["intensity_cumulative_weighted_sum"] / df["duration_weighted_sum"],
            0.0,
        )

    # --------------------------------------------------
    # 7. Clean up
    # --------------------------------------------------
    df = df.drop(columns=["unreal_sst"])

    # --------------------------------------------------
    # 8. Save
    # --------------------------------------------------
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"[INFO] Counterfactual data saved → {args.out}")
    print(f"[INFO] Rows: {len(df)}")


if __name__ == "__main__":
    main()
