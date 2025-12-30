# ============================================================
# A2: Temporal Structure Consistency Check
#
# Purpose
# -------
# This script is designed to visually examine whether the
# trained Temporal Fusion Transformer (TFT) model has learned
# the *temporal structure* of mangrove GPP time series.
#
# Specifically, it checks whether predicted GPP can reproduce:
#   (1) seasonal cycles (intra-annual variability), and
#   (2) short-term interannual continuity,
# when compared with observed GPP.
#
# This corresponds to evaluation task A2 in the model
# credibility framework of this study.
#
#
# Input
# -----
# results/predictions/factual_rolling_predictions.csv
#
# This CSV must contain at least the following columns:
#   - grid_id   : unique grid identifier
#   - year      : calendar year (integer)
#   - month     : calendar month (1–12)
#   - lat_c     : latitude of grid center (degrees)
#   - gpp_true  : observed monthly GPP (g C month^-1)
#   - gpp_pred  : model-predicted monthly GPP (g C month^-1)
#
# The file is typically produced by:
#   factual_rolling_predict.py
#
#
# Methodology
# ------------
# 1. Latitude band stratification
#    All grids are grouped into three latitude bands based on
#    absolute latitude:
#
#      - Low latitude  : |lat| < 10°
#      - Mid latitude  : 10° ≤ |lat| < 20°
#      - High latitude : |lat| ≥ 20°
#
#    This ensures spatial representativeness across the
#    global mangrove distribution.
#
# 2. Random grid selection (within each latitude band)
#    From each latitude band, ONE grid is randomly selected.
#
#    The randomness is controlled by a fixed random seed
#    (argument: --seed), which guarantees that:
#      - the same seed → the same selected grids
#      - different seeds → different but reproducible grids
#
#    IMPORTANT:
#    The seed here ONLY controls grid selection for plotting.
#    It does NOT affect:
#      - model training
#      - model parameters
#      - prediction values
#
# 3. Time window selection
#    For each selected grid:
#      - the data are sorted by (year, month)
#      - the FIRST `window` months (default: 24 months)
#        are plotted
#
#    This provides a clear two-year time span that is long
#    enough to show a full seasonal cycle while remaining
#    visually interpretable.
#
#    Note:
#    The script does NOT cherry-pick extreme events or
#    specific years; it always takes the earliest available
#    continuous period for each selected grid.
#
# 4. Visualization
#    For each latitude band, a time series plot is generated
#    showing:
#      - Observed GPP  (solid black line)
#      - Predicted GPP (blue dashed line)
#
#    The final figure contains:
#      - one subplot per latitude band
#      - latitude and grid_id annotated in titles
#      - a global title indicating the random seed used
#
#
# Output
# ------
# results/figures/A2_latband_timeseries.png
#
# This figure is intended for:
#   - qualitative evaluation of temporal realism
#   - inclusion in the main manuscript or supplementary material
#
#
# Interpretation (for manuscript)
# -------------------------------
# If the predicted GPP closely follows the observed seasonal
# cycles and maintains temporal continuity across months,
# this indicates that the model has learned meaningful
# temporal dependencies rather than producing noisy or
# memoryless predictions.
#
# This check complements quantitative metrics (A1) by
# providing an intuitive and transparent validation of
# temporal learning.
#
#
# Usage examples
# --------------
# Default (fixed seed, 24 months):
#   python -m code.evaluate.a2_latband_timeseries
#
# Specify a different random seed:
#   python -m code.evaluate.a2_latband_timeseries --seed 123
#
# Change time window length:
#   python -m code.evaluate.a2_latband_timeseries --window 36
#
# Reproducibility tip:
#   Run the script with multiple seeds (e.g., 1, 10, 42)
#   to confirm that conclusions do not depend on a single
#   grid selection.
#
# ============================================================


import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="results/predictions/factual_rolling_predictions.csv",
        help="Factual rolling prediction CSV",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for grid selection",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=24,
        help="Number of months to display",
    )
    args = parser.parse_args()

    # --------------------------------------------------
    # 1. load data
    # --------------------------------------------------
    df = pd.read_csv(args.csv)

    required = [
        "grid_id", "year", "month",
        "lat_c", "gpp_true", "gpp_pred"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    # sort for safety
    df = df.sort_values(["grid_id", "year", "month"]).reset_index(drop=True)

    # --------------------------------------------------
    # 2. define latitude bands
    # --------------------------------------------------
    grid_lat = (
        df.groupby("grid_id")["lat_c"]
        .first()
        .reset_index()
    )
    grid_lat["abs_lat"] = grid_lat["lat_c"].abs()
    
    lat_bands = {
        "Low latitude (|lat| < 10°)": grid_lat[grid_lat["abs_lat"] < 10],
        "Mid latitude (10° ≤ |lat| < 20°)": grid_lat[
            (grid_lat["abs_lat"] >= 10) & (grid_lat["abs_lat"] < 20)
        ],
        "High latitude (|lat| ≥ 20°)": grid_lat[grid_lat["abs_lat"] >= 20],
    }

    rng = np.random.RandomState(args.seed)

    selected = {}
    for band, sub in lat_bands.items():
        if len(sub) == 0:
            continue
        selected[band] = rng.choice(sub["grid_id"].values, size=1)[0]

    if len(selected) == 0:
        raise RuntimeError("No grid selected in any latitude band")

    # --------------------------------------------------
    # 3. plotting
    # --------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=len(selected),
        ncols=1,
        figsize=(10, 3.5 * len(selected)),
        sharex=False,
    )

    if len(selected) == 1:
        axes = [axes]

    for ax, (band, gid) in zip(axes, selected.items()):
        df_g = df[df["grid_id"] == gid].copy()
        df_g["time"] = pd.to_datetime(
            dict(year=df_g["year"], month=df_g["month"], day=1)
        )

        # take first window months
        df_g = df_g.iloc[: args.window]

        ax.plot(
            df_g["time"],
            df_g["gpp_true"],
            label="Observed GPP",
            color="black",
            linewidth=2,
        )
        ax.plot(
            df_g["time"],
            df_g["gpp_pred"],
            label="Predicted GPP",
            color="tab:blue",
            linestyle="--",
            linewidth=1.8,
        )

        lat_vals = df_g["lat_c"].unique()
        if len(lat_vals) != 1:
            raise RuntimeError(f"grid_id={grid_id} has inconsistent lat_c")
        lat_val = lat_vals[0]

        ax.set_title(
            f"{band} | grid_id={gid}, lat={lat_val:.2f}°"
        )
        ax.set_ylabel("Monthly GPP (g C month$^{-1}$)")
        ax.legend(frameon=False)

    axes[-1].set_xlabel("Time")
# ✅ 新增：整图标题 + seed
    fig.suptitle(
    f"A2: Temporal Structure Consistency (random seed = {args.seed})",
    fontsize=14
)

# 手动留出顶部空间，保证 suptitle 不被裁掉
    fig.subplots_adjust(top=0.88)


    #fig.tight_layout(rect=[0, 0, 1, 0.96])
   
    out_path = os.path.join(args.out_dir, "A2_latband_timeseries.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"[INFO] Saved → {out_path}")
    print("[INFO] Selected grids:", selected)


if __name__ == "__main__":
    main()
