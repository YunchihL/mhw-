# ============================================================
# A3: Inertness Check (Non-MHW vs MHW)
#
# Purpose
# -------
# This script checks whether the model behaves "inert" during
# non-event periods. For counterfactual simulations to be credible,
# the model should NOT generate spurious responses when no MHW occurs.
#
# Method
# ------
# 1) Read factual rolling predictions:
#      grid_id, year, month, gpp_true, gpp_pred
# 2) Join isMHW from the original data.csv on (grid_id, year, month)
# 3) Compute absolute percentage error:
#      abs_delta_pct = abs(gpp_pred - gpp_true) / abs(gpp_true) * 100
# 4) Compare distributions for:
#      - Non-MHW months (isMHW=0)
#      - MHW months     (isMHW=1)
#
# Output
# ------
# - results/tables/A3_inertness_summary.csv
# - results/figures/A3_inertness_abs_delta_pct.png
#
# Usage
# -----
# python -m code.evaluate.a3_inertness_check \
#   --pred results/predictions/factual_rolling_predictions.csv \
#   --data data/data.csv \
#   --out-dir results
#
# Optional:
#   --eps 1e-6            # avoid division by zero
#   --clip-pct 200        # cap extreme % values for plotting
#   --seed 42             # for reproducible bootstrap CI (optional)
# ============================================================

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def FATAL(msg: str):
    raise RuntimeError(f"\n[FATAL] {msg}\n")


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def percentile_ci(x: np.ndarray, q: float, n_boot: int = 2000, seed: int = 42):
    """
    Simple bootstrap CI for a quantile (e.g., median).
    Returns (estimate, lo, hi).
    """
    rng = np.random.RandomState(seed)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    est = np.percentile(x, q)
    if len(x) < 5:
        return (est, np.nan, np.nan)

    boots = []
    n = len(x)
    for _ in range(n_boot):
        samp = x[rng.randint(0, n, size=n)]
        boots.append(np.percentile(samp, q))
    lo, hi = np.percentile(np.array(boots), [2.5, 97.5])
    return (est, lo, hi)


def summarize_group(arr: np.ndarray, label: str, seed: int):
    arr = arr[~np.isnan(arr)]
    out = {
        "group": label,
        "n": int(len(arr)),
        "mean": float(np.mean(arr)) if len(arr) else np.nan,
        "std": float(np.std(arr)) if len(arr) else np.nan,
        "p50": float(np.percentile(arr, 50)) if len(arr) else np.nan,
        "p75": float(np.percentile(arr, 75)) if len(arr) else np.nan,
        "p90": float(np.percentile(arr, 90)) if len(arr) else np.nan,
        "p95": float(np.percentile(arr, 95)) if len(arr) else np.nan,
    }

    # Bootstrap CI for median (optional but persuasive)
    m, lo, hi = percentile_ci(arr, 50, n_boot=2000, seed=seed)
    out["p50_ci_lo"] = lo
    out["p50_ci_hi"] = hi
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred",
        type=str,
        default="results/predictions/factual_rolling_predictions.csv",
        help="Rolling factual predictions CSV",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/data.csv",
        help="Original data.csv that contains isMHW",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="Output root directory (tables/ + figures/ will be created under it)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="Small epsilon to avoid division by zero",
    )
    parser.add_argument(
        "--clip-pct",
        type=float,
        default=200.0,
        help="Clip abs_delta_pct for plotting (does not affect summary unless you want it to)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for bootstrap CI (reproducible)",
    )
    args = parser.parse_args()

    # -----------------------------
    # 1) Load predictions
    # -----------------------------
    if not os.path.exists(args.pred):
        FATAL(f"pred file not found: {args.pred}")
    pred = pd.read_csv(args.pred)

    need_pred = {"grid_id", "year", "month", "gpp_true", "gpp_pred"}
    miss = need_pred - set(pred.columns)
    if miss:
        FATAL(f"pred missing columns: {sorted(miss)}")

    # normalize key dtypes
    pred["grid_id"] = pred["grid_id"].astype(int)
    pred["year"] = pred["year"].astype(int)
    pred["month"] = pred["month"].astype(int)

    # -----------------------------
    # 2) Load data.csv for isMHW
    # -----------------------------
    if not os.path.exists(args.data):
        FATAL(f"data file not found: {args.data}")
    raw = pd.read_csv(args.data)

    need_raw = {"grid_id", "year", "month", "isMHW"}
    miss = need_raw - set(raw.columns)
    if miss:
        FATAL(f"data.csv missing columns: {sorted(miss)}")

    raw = raw[list(need_raw)].copy()
    raw["grid_id"] = raw["grid_id"].astype(int)
    raw["year"] = raw["year"].astype(int)
    raw["month"] = raw["month"].astype(int)
    raw["isMHW"] = raw["isMHW"].astype(int)

    # -----------------------------
    # 3) Merge on (grid_id, year, month)
    # -----------------------------
    df = pred.merge(
        raw,
        on=["grid_id", "year", "month"],
        how="left",
        validate="one_to_one",
    )

    if df["isMHW"].isna().any():
        n = int(df["isMHW"].isna().sum())
        FATAL(
            f"{n} rows cannot find isMHW after merge. "
            f"Check key alignment between pred and data.csv."
        )

    # -----------------------------
    # 4) Compute abs_delta_pct
    # -----------------------------
    # Use abs(gpp_true) in denominator to avoid sign issues in normalized mishaps (but your gpp_true should be >0)
    denom = np.maximum(np.abs(df["gpp_true"].values), args.eps)
    abs_delta_pct = np.abs(df["gpp_pred"].values - df["gpp_true"].values) / denom * 100.0
    df["abs_delta_pct"] = abs_delta_pct

    # -----------------------------
    # 5) Split groups
    # -----------------------------
    non = df.loc[df["isMHW"] == 0, "abs_delta_pct"].to_numpy()
    mhw = df.loc[df["isMHW"] == 1, "abs_delta_pct"].to_numpy()

    if len(non) == 0 or len(mhw) == 0:
        FATAL(f"Group size issue: nonMHW={len(non)}, MHW={len(mhw)}")

    # -----------------------------
    # 6) Summary table
    # -----------------------------
    rows = []
    rows.append(summarize_group(non, "nonMHW (isMHW=0)", seed=args.seed))
    rows.append(summarize_group(mhw, "MHW (isMHW=1)", seed=args.seed))

    # simple effect summaries
    med_non = rows[0]["p50"]
    med_mhw = rows[1]["p50"]
    rows.append({
        "group": "difference (MHW - nonMHW)",
        "n": np.nan,
        "mean": rows[1]["mean"] - rows[0]["mean"],
        "std": np.nan,
        "p50": med_mhw - med_non,
        "p75": np.nan,
        "p90": np.nan,
        "p95": np.nan,
        "p50_ci_lo": np.nan,
        "p50_ci_hi": np.nan,
    })
    rows.append({
        "group": "ratio (MHW / nonMHW) [median]",
        "n": np.nan,
        "mean": np.nan,
        "std": np.nan,
        "p50": (med_mhw / med_non) if (med_non and med_non > 0) else np.nan,
        "p75": np.nan,
        "p90": np.nan,
        "p95": np.nan,
        "p50_ci_lo": np.nan,
        "p50_ci_hi": np.nan,
    })

    summary = pd.DataFrame(rows)

    tables_dir = os.path.join(args.out_dir, "tables")
    figs_dir = os.path.join(args.out_dir, "figures")
    safe_mkdir(tables_dir)
    safe_mkdir(figs_dir)

    summary_path = os.path.join(tables_dir, "A3_inertness_summary.csv")
    summary.to_csv(summary_path, index=False)

    # # -----------------------------
    # # 7) Plot distributions
    # # -----------------------------
    # # clip only for plotting readability
    # non_plot = np.clip(non, 0, args.clip_pct)
    # mhw_plot = np.clip(mhw, 0, args.clip_pct)

    # plt.figure()
    # plt.hist(non_plot, bins=50, alpha=0.6, density=True, label="nonMHW (isMHW=0)")
    # plt.hist(mhw_plot, bins=50, alpha=0.6, density=True, label="MHW (isMHW=1)")
    # plt.xlabel(r"|ΔGPP%| = |(GPP_pred - GPP_true)/GPP_true| × 100 (%)")
    # plt.ylabel("Density")
    # plt.title("A3 Inertness Check: Error Distribution (nonMHW vs MHW)")
    # plt.legend(frameon=False)
    # fig_path = os.path.join(figs_dir, "A3_inertness_abs_delta_pct.png")
    # plt.tight_layout()
    # plt.savefig(fig_path, dpi=300)
    # plt.close()

    # -----------------------------
    # 7) Plot histogram as sample fraction (20% bins)
    # -----------------------------
    bins = np.arange(0, args.clip_pct + 20, 20)

    non_plot = np.clip(non, 0, args.clip_pct)
    mhw_plot = np.clip(mhw, 0, args.clip_pct)

    non_counts, _ = np.histogram(non_plot, bins=bins)
    mhw_counts, _ = np.histogram(mhw_plot, bins=bins)

    non_frac = non_counts / non_counts.sum()
    mhw_frac = mhw_counts / mhw_counts.sum()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    width = 0.35 * (bins[1] - bins[0])

    plt.figure(figsize=(8, 4))

    plt.bar(
        bin_centers - width / 2,
        non_frac,
        width=width,
        label="nonMHW (isMHW=0)",
    )
    plt.bar(
        bin_centers + width / 2,
        mhw_frac,
        width=width,
        label="MHW (isMHW=1)",
    )

    plt.xlabel(r"|ΔGPP| / GPP × 100 (%)")
    plt.ylabel("Fraction of samples")
    plt.title("A3 Inertness Check: Error Distribution (20% bins)")
    plt.xticks(bins)
    plt.legend(frameon=False)

    fig_path = os.path.join(figs_dir, "A3_inertness_abs_delta_pct.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    # -----------------------------
    # 7b) Boxplot (recommended by reviewers)
    # -----------------------------
    non_box = np.clip(non, 0, 100)
    mhw_box = np.clip(mhw, 0, 100)

    plt.figure(figsize=(5, 4))

    plt.boxplot(
        [non_box, mhw_box],
        labels=["nonMHW", "MHW"],
        showfliers=False,   # 不画极端离群点，避免视觉噪声
        widths=0.6,
    )

    plt.ylabel(r"|ΔGPP| / GPP × 100 (%)")
    plt.title("A3 Inertness Check: Error Distribution (Boxplot)")

    fig_path_box = os.path.join(figs_dir, "A3_inertness_abs_delta_pct_boxplot.png")
    plt.tight_layout()
    plt.savefig(fig_path_box, dpi=300)
    plt.close()

    print(f"[INFO] Saved figure → {fig_path_box}")


    # -----------------------------
    # 8) Print short report
    # -----------------------------
    print("\n========== A3: Inertness Check ==========")
    print(f"N (nonMHW) : {len(non)}")
    print(f"N (MHW)    : {len(mhw)}")
    print(f"Median |ΔGPP%| nonMHW: {rows[0]['p50']:.3f}% "
          f"(CI [{rows[0]['p50_ci_lo']:.3f}, {rows[0]['p50_ci_hi']:.3f}])")
    print(f"Median |ΔGPP%| MHW   : {rows[1]['p50']:.3f}% "
          f"(CI [{rows[1]['p50_ci_lo']:.3f}, {rows[1]['p50_ci_hi']:.3f}])")
    print(f"Median diff (MHW-non): {rows[2]['p50']:.3f}%")
    if not np.isnan(rows[3]["p50"]):
        print(f"Median ratio (MHW/non): {rows[3]['p50']:.3f}×")
    print("=========================================")
    print(f"[INFO] Saved table  → {summary_path}")
    print(f"[INFO] Saved figure → {fig_path}\n")


if __name__ == "__main__":
    main()
