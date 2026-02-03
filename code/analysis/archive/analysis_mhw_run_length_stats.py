# ============================================================
# Step 5.5: MHW run-length structure at monthly scale
#
# Purpose:
#   Quantify whether MHWs appear as
#   - single-month pulses
#   - or multi-month continuous episodes
#
# Input:
#   data.csv with columns:
#     grid_id, year, month, isMHW
#
# Output:
#   - mhw_run_length_summary.csv
#   - mhw_run_length_distribution.csv
#   - mhw_run_length_by_grid.csv
# ============================================================

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        required=True,
        help="Path to data.csv (must contain grid_id, year, month, isMHW)",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Directory to save results",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to save a run-length histogram",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    df = pd.read_csv(args.data)

    required_cols = ["grid_id", "year", "month", "isMHW"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df = df.sort_values(["grid_id", "year", "month"]).reset_index(drop=True)

    # --------------------------------------------------------
    # Identify run lengths (per grid)
    # --------------------------------------------------------
    run_records = []

    for gid, g in df.groupby("grid_id"):
        g = g.reset_index(drop=True)

        run_len = 0
        for i, row in g.iterrows():
            if row["isMHW"] == 1:
                run_len += 1
            else:
                if run_len > 0:
                    run_records.append(
                        {
                            "grid_id": gid,
                            "run_length": run_len,
                        }
                    )
                    run_len = 0

        # catch run reaching the end
        if run_len > 0:
            run_records.append(
                {
                    "grid_id": gid,
                    "run_length": run_len,
                }
            )

    df_runs = pd.DataFrame(run_records)

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------
    total_runs = len(df_runs)
    single_runs = (df_runs["run_length"] == 1).sum()
    multi_runs = (df_runs["run_length"] >= 2).sum()

    summary = pd.DataFrame(
        {
            "total_runs": [total_runs],
            "single_month_runs": [single_runs],
            "multi_month_runs": [multi_runs],
            "single_month_ratio": [single_runs / total_runs if total_runs > 0 else 0],
            "multi_month_ratio": [multi_runs / total_runs if total_runs > 0 else 0],
            "max_run_length": [df_runs["run_length"].max()],
            "median_run_length": [df_runs["run_length"].median()],
            "mean_run_length": [df_runs["run_length"].mean()],
        }
    )

    # --------------------------------------------------------
    # Run-length distribution
    # --------------------------------------------------------
    dist = (
    df_runs["run_length"]
    .value_counts()
    .sort_index()
    .reset_index()
    )
    dist.columns = ["run_length", "count"]

    # --------------------------------------------------------
    # Per-grid maximum run length
    # --------------------------------------------------------
    by_grid = (
        df_runs.groupby("grid_id")["run_length"]
        .max()
        .reset_index()
        .rename(columns={"run_length": "max_run_length"})
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------
    summary_path = os.path.join(args.outdir, "mhw_run_length_summary.csv")
    dist_path = os.path.join(args.outdir, "mhw_run_length_distribution.csv")
    grid_path = os.path.join(args.outdir, "mhw_run_length_by_grid.csv")

    summary.to_csv(summary_path, index=False)
    dist.to_csv(dist_path, index=False)
    by_grid.to_csv(grid_path, index=False)

    print("============================================================")
    print("[MHW run-length analysis completed]")
    print("------------------------------------------------------------")
    print(summary)
    print("------------------------------------------------------------")
    print(f"[OUT] {summary_path}")
    print(f"[OUT] {dist_path}")
    print(f"[OUT] {grid_path}")
    print("============================================================")

    # --------------------------------------------------------
    # Optional plot
    # --------------------------------------------------------
    if args.plot:
        plt.figure(figsize=(6, 4))
        plt.bar(dist["run_length"], dist["count"])
        plt.xlabel("Consecutive MHW months (run length)")
        plt.ylabel("Number of runs")
        plt.title("Distribution of monthly MHW run lengths")
        plt.tight_layout()

        plot_path = os.path.join(args.outdir, "mhw_run_length_distribution.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"[OUT] {plot_path}")


if __name__ == "__main__":
    main()
