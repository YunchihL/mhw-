# ============================================================
# STEP 7A: Spatial heterogeneity (NEG vs OTHERS)
#   - Primary: relative sensitivity (delta_pct, %)
#   - Secondary: absolute magnitude (delta_abs)
# Outputs:
#   - step7a_spatial_summary.csv
#   - step7a_grid_level_metrics.csv
#   - boxplots: lat/lon/area/delta_abs/delta_pct
#   - lon-lat scatter colored by group
# ============================================================

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# helpers
# ----------------------------
def qstats(x: pd.Series) -> dict:
    x = x.dropna().values
    if len(x) == 0:
        return dict(n=0, mean=np.nan, median=np.nan, p25=np.nan, p75=np.nan)
    return dict(
        n=int(len(x)),
        mean=float(np.mean(x)),
        median=float(np.median(x)),
        p25=float(np.percentile(x, 25)),
        p75=float(np.percentile(x, 75)),
    )


def safe_mannwhitneyu(x, y):
    """
    Optional: do a Mann–Whitney U test if scipy is available.
    If not, return NaN.
    """
    try:
        from scipy.stats import mannwhitneyu
        x = pd.Series(x).dropna().values
        y = pd.Series(y).dropna().values
        if len(x) == 0 or len(y) == 0:
            return np.nan
        return float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def boxplot_two_groups(df, value_col, outpath, title):
    """
    Simple boxplot for NEG vs OTHERS.
    """
    neg = df.loc[df["group"] == "NEG", value_col].dropna().values
    oth = df.loc[df["group"] == "OTHERS", value_col].dropna().values

    plt.figure()
    plt.boxplot([neg, oth], labels=["NEG", "OTHERS"], showfliers=True)
    plt.title(title)
    plt.ylabel(value_col)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# ----------------------------
# main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, help="grid_multiyear_summary.csv")
    parser.add_argument("--factual", required=True, help="factual rolling predictions CSV")
    parser.add_argument("--counterfactual", required=True, help="counterfactual rolling predictions CSV")
    parser.add_argument("--data", required=True, help="data.csv containing lon_c, lat_c, mangrove_area")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("============================================================")
    print("[STEP 7A] Spatial heterogeneity: NEG vs OTHERS")
    print("Primary: relative sensitivity (delta_pct, %)")
    print("Secondary: absolute magnitude (delta_abs)")
    print("------------------------------------------------------------")
    print("Inputs:")
    print(f"  summary:       {args.summary}")
    print(f"  factual:       {args.factual}")
    print(f"  counterfactual:{args.counterfactual}")
    print(f"  data:          {args.data}")
    print(f"  outdir:        {args.outdir}")
    print("============================================================")

    # --- load ---
    summ = pd.read_csv(args.summary)
    fac  = pd.read_csv(args.factual)
    cf   = pd.read_csv(args.counterfactual)
    dat  = pd.read_csv(args.data)

    # --- ensure keys ---
    key = ["grid_id", "year", "month"]
    for k in key:
        if k not in fac.columns or k not in cf.columns or k not in dat.columns:
            raise ValueError(f"Missing key column '{k}' in one of the inputs.")

    # --- group label from summary ---
    # Prefer explicit "group" if present.
    if "group" in summ.columns:
        group_map = summ[["grid_id", "group"]].drop_duplicates()
    else:
        # Fallback inference:
        # 1) if mean_delta_gpp_year exists: NEG if < 0
        # 2) else if n_year_neg/n_year_pos exists: NEG if n_year_neg >= n_year_pos
        if "mean_delta_gpp_year" in summ.columns:
            tmp = summ[["grid_id", "mean_delta_gpp_year"]].drop_duplicates()
            tmp["group"] = np.where(tmp["mean_delta_gpp_year"] < 0, "NEG", "OTHERS")
            group_map = tmp[["grid_id", "group"]]
        elif ("n_year_neg" in summ.columns) and ("n_year_pos" in summ.columns):
            tmp = summ[["grid_id", "n_year_neg", "n_year_pos"]].drop_duplicates()
            tmp["group"] = np.where(tmp["n_year_neg"] >= tmp["n_year_pos"], "NEG", "OTHERS")
            group_map = tmp[["grid_id", "group"]]
        else:
            raise ValueError(
                "Cannot infer group. summary must contain either "
                "'group' or 'mean_delta_gpp_year' or ('n_year_neg' and 'n_year_pos')."
            )

    # --- static spatial vars (lon/lat/area) ---
    static_cols = ["grid_id", "lon_c", "lat_c", "mangrove_area"]
    for c in static_cols:
        if c not in dat.columns:
            raise ValueError(f"data.csv missing required column: {c}")
    static = dat[static_cols].drop_duplicates(subset=["grid_id"]).copy()

    # --- build month-level delta from predictions ---
    if "gpp_pred" not in fac.columns or "gpp_pred" not in cf.columns:
        raise ValueError("Both factual and counterfactual prediction files must contain column 'gpp_pred'.")
    if "gpp_true" not in fac.columns:
        raise ValueError("factual prediction file must contain column 'gpp_true' (for baseline GPP).")

    df = fac[key + ["gpp_pred", "gpp_true"]].merge(
        cf[key + ["gpp_pred"]],
        on=key,
        suffixes=("_factual", "_cf"),
        how="inner"
    )
    df["delta_abs_month"] = df["gpp_pred_cf"] - df["gpp_pred_factual"]

    # --- baseline GPP per grid (median of factual gpp_true across all months) ---
    baseline = (
        df.groupby("grid_id")["gpp_true"]
        .median()
        .rename("baseline_gpp_true_median_month")
        .reset_index()
    )
    # annualize baseline (approx): 12 * median monthly
    baseline["baseline_gpp_true_median_year"] = baseline["baseline_gpp_true_median_month"] * 12.0

    # --- grid-level delta metrics ---
    # Absolute: mean month delta, and annualized mean (x12)
    grid_delta = df.groupby("grid_id")["delta_abs_month"].mean().rename("delta_abs_mean_month").reset_index()
    grid_delta["delta_abs_mean_year_approx"] = grid_delta["delta_abs_mean_month"] * 12.0

    # Relative sensitivity: delta_pct using baseline (avoid dividing by tiny numbers)
    merged = grid_delta.merge(baseline, on="grid_id", how="left")

    # epsilon to stabilize %: use 5th percentile of baseline monthly (positive values)
    pos_base = merged["baseline_gpp_true_median_month"].replace([np.inf, -np.inf], np.nan).dropna()
    pos_base = pos_base[pos_base > 0]
    eps = float(np.percentile(pos_base, 5)) if len(pos_base) else 1.0

    merged["delta_pct_mean_month"] = merged["delta_abs_mean_month"] / (merged["baseline_gpp_true_median_month"] + eps) * 100.0
    merged["delta_pct_mean_year_approx"] = merged["delta_abs_mean_year_approx"] / (merged["baseline_gpp_true_median_year"] + eps * 12.0) * 100.0

    # --- bring in annual delta from summary if available (secondary absolute magnitude) ---
    if "mean_delta_gpp_year" in summ.columns:
        annual_from_summary = summ[["grid_id", "mean_delta_gpp_year"]].drop_duplicates()
        merged = merged.merge(annual_from_summary, on="grid_id", how="left")
    else:
        merged["mean_delta_gpp_year"] = np.nan

    # --- attach group + spatial ---
    out = merged.merge(group_map, on="grid_id", how="left").merge(static, on="grid_id", how="left")

    # sanity: any missing groups?
    missing_g = out["group"].isna().sum()
    if missing_g > 0:
        print(f"[WARN] {missing_g} grids have missing group label after merge. They will be dropped.")
        out = out.dropna(subset=["group"]).copy()

    # --- write grid-level metrics table ---
    grid_level_path = os.path.join(args.outdir, "step7a_grid_level_metrics.csv")
    out.to_csv(grid_level_path, index=False)

    # --- summarize by group (core table) ---
    metrics = [
        "lat_c", "lon_c", "mangrove_area",
        "baseline_gpp_true_median_month", "baseline_gpp_true_median_year",
        "delta_abs_mean_month", "delta_abs_mean_year_approx",
        "delta_pct_mean_month", "delta_pct_mean_year_approx",
        "mean_delta_gpp_year",
    ]

    rows = []
    for m in metrics:
        for grp in ["NEG", "OTHERS"]:
            s = out.loc[out["group"] == grp, m]
            st = qstats(s)
            st.update(dict(metric=m, group=grp))
            rows.append(st)

        # optional p-value
        p = safe_mannwhitneyu(
            out.loc[out["group"] == "NEG", m],
            out.loc[out["group"] == "OTHERS", m],
        )
        rows.append(dict(metric=m, group="MannWhitneyU_p", n=np.nan, mean=np.nan, median=np.nan, p25=np.nan, p75=np.nan, pvalue=p))

    summary_df = pd.DataFrame(rows)

    summary_path = os.path.join(args.outdir, "step7a_spatial_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # --- plots ---
    plot_dir = os.path.join(args.outdir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # boxplots: lat/lon/area + delta_abs + delta_pct
    boxplot_two_groups(out, "lat_c", os.path.join(plot_dir, "step7a_box_lat.png"), "STEP 7A: Latitude by group")
    boxplot_two_groups(out, "lon_c", os.path.join(plot_dir, "step7a_box_lon.png"), "STEP 7A: Longitude by group")
    boxplot_two_groups(out, "mangrove_area", os.path.join(plot_dir, "step7a_box_area.png"), "STEP 7A: Mangrove area by group")

    boxplot_two_groups(out, "delta_abs_mean_year_approx", os.path.join(plot_dir, "step7a_box_delta_abs_year.png"),
                       "STEP 7A: Absolute delta (approx annual) by group")
    boxplot_two_groups(out, "delta_pct_mean_year_approx", os.path.join(plot_dir, "step7a_box_delta_pct_year.png"),
                       "STEP 7A: Relative delta (%) (approx annual) by group")

    # lon-lat scatter
    plt.figure()
    neg = out[out["group"] == "NEG"]
    oth = out[out["group"] == "OTHERS"]
    plt.scatter(neg["lon_c"], neg["lat_c"], label="NEG", s=25)
    plt.scatter(oth["lon_c"], oth["lat_c"], label="OTHERS", s=25)
    plt.title("STEP 7A: Spatial distribution (lon vs lat)")
    plt.xlabel("lon_c")
    plt.ylabel("lat_c")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "step7a_scatter_lon_lat.png"), dpi=200)
    plt.close()

    print("------------------------------------------------------------")
    print("[STEP 7A DONE] Key outputs:")
    print(f"  - {grid_level_path}")
    print(f"  - {summary_path}")
    print(f"  - {plot_dir}/step7a_box_lat.png")
    print(f"  - {plot_dir}/step7a_box_lon.png")
    print(f"  - {plot_dir}/step7a_scatter_lon_lat.png")
    print("Notes:")
    print(f"  - eps used for delta_pct stabilization = {eps:.6e}")
    print("============================================================")


if __name__ == "__main__":
    main()
