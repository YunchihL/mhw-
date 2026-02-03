# ============================================================
# STEP 7C (v1): Background state × MHW structure (event months)
#   - Background state: "high vs low" based on within-grid gpp_true median
#   - Structure: all pairwise 2×2 bins among duration/intensity/density (median split)
#   - Outcome:
#       loss_abs = factual_pred - cf_pred
#       loss_pct = loss_abs / (|factual_pred| + eps) * 100
#   - Optional stratification: NEG vs OTHERS from grid_multiyear_summary.csv
#
# Outputs:
#   - step7c_event_rows_with_states.csv
#   - step7c_bins_background_x_structure.csv
#   - step7c_bins_background_x_structure_by_group.csv (if summary provided)
# ============================================================

import argparse
import os
import numpy as np
import pandas as pd


STRUCT_VARS = [
    "duration_weighted_sum",
    "intensity_cumulative_weighted_sum",
    "intensity_density",
]

KEYS = ["grid_id", "year", "month"]


def infer_group_map(summary_path: str | None):
    if summary_path is None:
        return None

    summ = pd.read_csv(summary_path)

    # use explicit "group" if present
    if "group" in summ.columns:
        gm = summ[["grid_id", "group"]].drop_duplicates()
        return gm

    # fallback: mean_delta_gpp_year
    if "mean_delta_gpp_year" in summ.columns:
        tmp = summ[["grid_id", "mean_delta_gpp_year"]].drop_duplicates()
        tmp["group"] = np.where(tmp["mean_delta_gpp_year"] < 0, "NEG", "OTHERS")
        return tmp[["grid_id", "group"]]

    # fallback: n_year_neg/n_year_pos
    if ("n_year_neg" in summ.columns) and ("n_year_pos" in summ.columns):
        tmp = summ[["grid_id", "n_year_neg", "n_year_pos"]].drop_duplicates()
        tmp["group"] = np.where(tmp["n_year_neg"] >= tmp["n_year_pos"], "NEG", "OTHERS")
        return tmp[["grid_id", "group"]]

    raise ValueError(
        "Cannot infer group from summary. Provide 'group' or "
        "'mean_delta_gpp_year' or ('n_year_neg' and 'n_year_pos')."
    )


def load_merge(factual_path, counterfactual_path, data_path):
    fac = pd.read_csv(factual_path)
    cf = pd.read_csv(counterfactual_path)
    dat = pd.read_csv(data_path)

    # basic checks
    for k in KEYS:
        if k not in fac.columns or k not in cf.columns or k not in dat.columns:
            raise ValueError(f"Missing key '{k}' in one of inputs.")

    if "gpp_pred" not in fac.columns or "gpp_pred" not in cf.columns:
        raise ValueError("Both factual and counterfactual predictions must contain 'gpp_pred'.")

    if "gpp_true" not in fac.columns:
        raise ValueError("Factual predictions must contain 'gpp_true' for background state definition.")

    need_cols = ["isMHW"] + STRUCT_VARS
    for c in need_cols:
        if c not in dat.columns:
            raise ValueError(f"data.csv missing required column: {c}")

    df = fac[KEYS + ["gpp_pred", "gpp_true"]].merge(
        cf[KEYS + ["gpp_pred"]],
        on=KEYS,
        suffixes=("_factual", "_cf"),
        how="inner",
    ).merge(
        dat[KEYS + need_cols],
        on=KEYS,
        how="left",
    )

    # outcome aligned to hazard view
    df["loss_abs"] = df["gpp_pred_factual"] - df["gpp_pred_cf"]

    # eps stabilization using p05 of |factual_pred|
    eps = float(np.percentile(np.abs(df["gpp_pred_factual"]), 5))
    df["loss_pct"] = df["loss_abs"] / (np.abs(df["gpp_pred_factual"]) + eps) * 100.0

    return df, eps


def add_background_state(df: pd.DataFrame):
    """
    Background state defined by within-grid median of gpp_true (across all months).
    high if gpp_true >= grid_median else low
    """
    gmed = df.groupby("grid_id")["gpp_true"].median().rename("gpp_true_grid_median").reset_index()
    out = df.merge(gmed, on="grid_id", how="left")
    out["bg_gpp_state"] = np.where(out["gpp_true"] >= out["gpp_true_grid_median"], "high", "low")
    return out


def median_thresholds(df_evt: pd.DataFrame, vars_list):
    th = {}
    for v in vars_list:
        th[v] = float(df_evt[v].median())
    return th


def bin2(sub: pd.DataFrame, x: str, y: str, th: dict):
    """
    2×2 bins for variables x,y using thresholds in th.
    """
    out = sub.copy()
    out["x_level"] = np.where(out[x] >= th[x], "high", "low")
    out["y_level"] = np.where(out[y] >= th[y], "high", "low")
    return out


def summarize_bins(df_binned: pd.DataFrame, x: str, y: str, extra_group_cols=None):
    """
    Summarize loss_pct and damage_ratio within each bin (and optionally extra grouping).
    """
    if extra_group_cols is None:
        extra_group_cols = []

    group_cols = extra_group_cols + ["bg_gpp_state", "x_level", "y_level"]

    def _agg(g):
        return pd.Series({
            "n": len(g),
            "loss_pct_median": g["loss_pct"].median(),
            "loss_pct_p25": g["loss_pct"].quantile(0.25),
            "loss_pct_p75": g["loss_pct"].quantile(0.75),
            "damage_ratio": (g["loss_abs"] > 0).mean(),
        })

    res = df_binned.groupby(group_cols, dropna=True).apply(_agg).reset_index()
    res.insert(0, "x", x)
    res.insert(1, "y", y)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factual", required=True)
    parser.add_argument("--counterfactual", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--summary", default=None, help="grid_multiyear_summary.csv (optional, for NEG/OTHERS stratification)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("============================================================")
    print("[STEP 7C] Background state × MHW structure (event months)")
    print("Background: within-grid median split using gpp_true (high/low)")
    print("Structure bins: median split, all pairwise combinations")
    print("Outcome: loss_abs = factual_pred - cf_pred; loss_pct stabilized by eps")
    print("------------------------------------------------------------")
    print(f"factual:        {args.factual}")
    print(f"counterfactual: {args.counterfactual}")
    print(f"data:           {args.data}")
    print(f"summary(opt):   {args.summary}")
    print(f"outdir:         {args.outdir}")
    print("============================================================")

    df, eps = load_merge(args.factual, args.counterfactual, args.data)
    df = add_background_state(df)

    # event months only
    df_evt = df[df["isMHW"] == 1].reset_index(drop=True)

    print(f"[STEP 7C] Event months n = {len(df_evt)}")
    print(f"[STEP 7C] eps (p05 of |factual_pred|) = {eps:.6e}")

    # thresholds for structure computed within event months
    th = median_thresholds(df_evt, STRUCT_VARS)
    print("[STEP 7C] Structure thresholds (median in event months):")
    for k, v in th.items():
        print(f"  - {k}: {v:.6f}")

    # optional NEG/OTHERS
    group_map = infer_group_map(args.summary) if args.summary else None
    if group_map is not None:
        df_evt = df_evt.merge(group_map, on="grid_id", how="left")
        df_evt["group"] = df_evt["group"].fillna("UNKNOWN")
        extra_cols = ["group"]
        print("[STEP 7C] Group stratification enabled (NEG/OTHERS).")
    else:
        extra_cols = []

    # save event rows (for reproducibility)
    rows_out = os.path.join(args.outdir, "step7c_event_rows_with_states.csv")
    keep_cols = KEYS + ["isMHW", "bg_gpp_state", "gpp_true", "gpp_true_grid_median",
                        "gpp_pred_factual", "gpp_pred_cf", "loss_abs", "loss_pct"] + STRUCT_VARS + extra_cols
    df_evt[keep_cols].to_csv(rows_out, index=False)

    # pairwise bins
    all_results = []
    all_results_by_group = []

    vars_list = STRUCT_VARS
    for i in range(len(vars_list)):
        for j in range(i + 1, len(vars_list)):
            x = vars_list[i]
            y = vars_list[j]
            b = bin2(df_evt, x, y, th)

            # overall (background only)
            res = summarize_bins(b, x, y, extra_group_cols=[])
            all_results.append(res)

            # background + NEG/OTHERS if available
            if group_map is not None:
                res_g = summarize_bins(b, x, y, extra_group_cols=["group"])
                all_results_by_group.append(res_g)

    out1 = pd.concat(all_results, ignore_index=True)
    out_path1 = os.path.join(args.outdir, "step7c_bins_background_x_structure.csv")
    out1.to_csv(out_path1, index=False)

    if group_map is not None:
        out2 = pd.concat(all_results_by_group, ignore_index=True)
        out_path2 = os.path.join(args.outdir, "step7c_bins_background_x_structure_by_group.csv")
        out2.to_csv(out_path2, index=False)

    print("------------------------------------------------------------")
    print("[STEP 7C DONE] Outputs:")
    print(f"  - {rows_out}")
    print(f"  - {out_path1}")
    if group_map is not None:
        print(f"  - {out_path2}")
    print("============================================================")


if __name__ == "__main__":
    main()
