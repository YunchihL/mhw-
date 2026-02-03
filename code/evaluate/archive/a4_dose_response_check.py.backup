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
    df = df.dropna(subset=["intensity_density", "abs_delta_gpp_pct"])

    # --------------------------------------------------
    # 5. LOWESS smoothing
    # --------------------------------------------------
    lowess_curve = lowess(
        df["abs_delta_gpp_pct"],
        df["intensity_density"],
        frac=args.frac,
        return_sorted=True
    )

    # # --------------------------------------------------
    # # 6. plotting
    # # --------------------------------------------------
    # fig_dir = os.path.join(args.out_dir, "figures")
    # tab_dir = os.path.join(args.out_dir, "tables")
    # os.makedirs(fig_dir, exist_ok=True)
    # os.makedirs(tab_dir, exist_ok=True)

    # plt.figure(figsize=(7, 5))
    # plt.scatter(
    #     df["intensity_density"],
    #     df["abs_delta_gpp_pct"],
    #     s=10,
    #     alpha=0.3,
    #     label="MHW months"
    # )
    # plt.plot(
    #     lowess_curve[:, 0],
    #     lowess_curve[:, 1],
    #     color="red",
    #     linewidth=2,
    #     label="LOWESS trend"
    # )

    # plt.xlabel("MHW intensity density")
    # plt.ylabel("|ΔGPP| (%)")
    # plt.title("A4: Dose–Response Check (MHW months only)")
    # plt.legend(frameon=False)
    # plt.tight_layout()

    # fig_path = os.path.join(fig_dir, "A4_dose_response_scatter.png")
    # plt.savefig(fig_path, dpi=300)
    # plt.close()

    # --------------------------------------------------
    # 6. plotting
    # --------------------------------------------------
    fig_dir = os.path.join(args.out_dir, "figures")
    tab_dir = os.path.join(args.out_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

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
        lowess_curve[:, 0],
        lowess_curve[:, 1],
        color="red",
        linewidth=2,
        label="LOWESS trend"
    )

    plt.xlabel("MHW intensity density")
    plt.ylabel("|ΔGPP| (%)")

    # ✅ 关键修改：固定纵轴范围
    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 101, 20))

    plt.title("A4: Dose–Response Check (MHW months only)")
    plt.legend(frameon=False)

    plt.tight_layout()

    fig_path = os.path.join(fig_dir, "A4_dose_response_scatter.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()


    # --------------------------------------------------
    # 7. summary table
    # --------------------------------------------------
    summary = (
        df["abs_delta_gpp_pct"]
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        .to_frame(name="|ΔGPP| (%)")
    )

    tab_path = os.path.join(tab_dir, "A4_dose_response_summary.csv")
    summary.to_csv(tab_path)

    # --------------------------------------------------
    # 8. log
    # --------------------------------------------------
    print("\n========== A4: Dose–Response Check ==========")
    print(f"N (MHW months): {len(df)}")
    print(f"Median |ΔGPP| (%): {summary.loc['50%', '|ΔGPP| (%)']:.2f}")
    print("=============================================")
    print(f"[INFO] Saved figure → {fig_path}")
    print(f"[INFO] Saved table  → {tab_path}")


if __name__ == "__main__":
    main()
