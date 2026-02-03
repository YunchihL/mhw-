# ============================================================
# analysis_align_pretrial_metric.py
#   这个脚本是为了查看预试验的结果和现在的正式实验的结果是否冲突
# ------------------------------------------------------------
# Purpose:
#   Align current TFT counterfactual results to the same metric
#   used in the "pretrial" logic:
#       loss = factual_pred - counterfactual_pred
#   and analyze dose-response within MHW months (isMHW=1):
#       loss_pct vs intensity_density / duration_weighted_sum /
#                  intensity_cumulative_weighted_sum
#
# Outputs (in --outdir):
#   - aligned_event_month_rows.csv
#   - dose_response_summary_event_months.csv
#   - dose_response_correlations_event_months.csv
#   - scatter_loss_pct_vs_*.png (3 figures)
#   - summary_pretrial_alignment.txt
# ============================================================

import os
import argparse
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd

# optional (but usually available)
try:
    import matplotlib.pyplot as plt
except Exception as e:
    plt = None

# optional stats packages
try:
    from scipy.stats import spearmanr, pearsonr
except Exception:
    spearmanr = None
    pearsonr = None

try:
    import statsmodels.api as sm
except Exception:
    sm = None


def _pick_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Missing required column. Tried: {candidates}. Available: {list(df.columns)}")
    return ""


def _ensure_str(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = df[col].astype(str)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Returns:
      spearman_r, spearman_p, pearson_r, pearson_p
    If scipy not available, returns nan.
    """
    if spearmanr is None or pearsonr is None:
        return (np.nan, np.nan, np.nan, np.nan)

    # drop nan pairwise
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return (np.nan, np.nan, np.nan, np.nan)

    sr, sp = spearmanr(x[m], y[m])
    pr, pp = pearsonr(x[m], y[m])
    return (float(sr), float(sp), float(pr), float(pp))


def _ols(y: np.ndarray, x: np.ndarray):
    """
    OLS: y = a + b*x
    Returns dict with coef, pvalue, r2, n
    """
    if sm is None:
        return {"n": int(np.isfinite(y).sum()), "coef": np.nan, "p": np.nan, "r2": np.nan}

    m = np.isfinite(y) & np.isfinite(x)
    if m.sum() < 10:
        return {"n": int(m.sum()), "coef": np.nan, "p": np.nan, "r2": np.nan}

    X = sm.add_constant(x[m])
    model = sm.OLS(y[m], X).fit()
    return {
        "n": int(m.sum()),
        "coef": float(model.params[1]),
        "p": float(model.pvalues[1]),
        "r2": float(model.rsquared),
        "intercept": float(model.params[0]),
    }


def _make_scatter(outpath: str, x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str):
    if plt is None:
        return

    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return

    plt.figure()
    plt.scatter(x[m], y[m], s=8, alpha=0.35)

    # simple fit line (numpy polyfit) for visualization
    try:
        b1, b0 = np.polyfit(x[m], y[m], 1)
        xs = np.linspace(np.nanmin(x[m]), np.nanmax(x[m]), 100)
        ys = b1 * xs + b0
        plt.plot(xs, ys)
    except Exception:
        pass

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factual", required=True, help="factual predictions csv (must include gpp_pred)")
    parser.add_argument("--counterfactual", required=True, help="counterfactual predictions csv (must include gpp_pred)")
    parser.add_argument("--data", required=True, help="data.csv with isMHW + mhw structure variables")
    parser.add_argument("--outdir", required=True, help="output directory")
    parser.add_argument("--eps_mode", default="p05", choices=["p05", "1e-6"], help="epsilon for pct denominator")
    parser.add_argument("--min_abs_factual", type=float, default=0.0, help="optional: filter tiny factual_pred")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ----------------------------
    # Load
    # ----------------------------
    factual = pd.read_csv(args.factual)
    cf = pd.read_csv(args.counterfactual)
    data = pd.read_csv(args.data)

    # ----------------------------
    # Identify keys
    # ----------------------------
    gid = _pick_col(factual, ["grid_id", "grid", "gid"])
    year = _pick_col(factual, ["year"])
    month = _pick_col(factual, ["month"])

    pred_col_f = _pick_col(factual, ["gpp_pred", "pred", "y_pred"])
    pred_col_cf = _pick_col(cf, ["gpp_pred", "pred", "y_pred"])

    true_col = _pick_col(factual, ["gpp_true", "true", "y_true"], required=False)

    # ensure merge types consistent
    _ensure_str(factual, gid)
    _ensure_str(cf, gid)
    _ensure_str(data, "grid_id")  # your data.csv uses grid_id

    # ----------------------------
    # Merge factual & cf on (grid_id, year, month)
    # ----------------------------
    f2 = factual[[gid, year, month, pred_col_f] + ([true_col] if true_col else [])].copy()
    f2 = f2.rename(columns={gid: "grid_id", pred_col_f: "factual_pred"})
    if true_col:
        f2 = f2.rename(columns={true_col: "gpp_true"})

    cf2 = cf[[gid, year, month, pred_col_cf]].copy()
    cf2 = cf2.rename(columns={gid: "grid_id", pred_col_cf: "cf_pred"})

    merged = f2.merge(cf2, on=["grid_id", year, month], how="inner", validate="one_to_one")

    # Merge with data.csv to get isMHW + mhw structure variables
    # Required in your project:
    # - isMHW
    # - duration_weighted_sum
    # - intensity_cumulative_weighted_sum
    # - intensity_density
    need_data_cols = ["grid_id", "year", "month", "isMHW",
                      "duration_weighted_sum", "intensity_cumulative_weighted_sum", "intensity_density"]
    missing = [c for c in need_data_cols if c not in data.columns]
    if missing:
        raise ValueError(f"data.csv missing required columns: {missing}\nAvailable: {list(data.columns)}")

    d2 = data[need_data_cols].copy()
    d2["grid_id"] = d2["grid_id"].astype(str)

    merged = merged.merge(d2, on=["grid_id", "year", "month"], how="left", validate="one_to_one")
    if merged["isMHW"].isna().any():
        n_na = int(merged["isMHW"].isna().sum())
        warnings.warn(f"[WARN] {n_na} rows have missing isMHW after merge. Check keys alignment.")

    # ----------------------------
    # Define "pretrial-aligned" loss
    # ----------------------------
    # loss > 0 means: factual_pred > cf_pred, i.e., removing MHW increases GPP (damage case)
    merged["loss_abs"] = merged["factual_pred"] - merged["cf_pred"]

    # denominator epsilon
    factual_pos = merged["factual_pred"].to_numpy()
    if args.eps_mode == "p05":
        # robust epsilon = 5th percentile of positive factual_pred
        pos = factual_pos[np.isfinite(factual_pos) & (factual_pos > 0)]
        eps = float(np.percentile(pos, 5)) if pos.size > 0 else 1e-6
    else:
        eps = 1e-6

    merged["loss_pct"] = merged["loss_abs"] / (merged["factual_pred"].abs() + eps) * 100.0

    # optional filter for very small factual_pred (rare in your case)
    if args.min_abs_factual > 0:
        merged = merged[merged["factual_pred"].abs() >= args.min_abs_factual].copy()

    # ----------------------------
    # Focus on MHW months (isMHW == 1)
    # ----------------------------
    merged["isMHW"] = pd.to_numeric(merged["isMHW"], errors="coerce")
    event = merged.loc[merged["isMHW"] == 1].copy()

    # Save aligned event rows (for debugging / future use)
    event_out = os.path.join(args.outdir, "aligned_event_month_rows.csv")
    event.to_csv(event_out, index=False)

    # ----------------------------
    # Summary stats (event months)
    # ----------------------------
    def desc(s: pd.Series):
        s = pd.to_numeric(s, errors="coerce")
        return {
            "n": int(s.notna().sum()),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "p25": float(s.quantile(0.25)),
            "p75": float(s.quantile(0.75)),
            "p05": float(s.quantile(0.05)),
            "p95": float(s.quantile(0.95)),
        }

    summary = {
        "loss_abs": desc(event["loss_abs"]),
        "loss_pct": desc(event["loss_pct"]),
        "duration_weighted_sum": desc(event["duration_weighted_sum"]),
        "intensity_cumulative_weighted_sum": desc(event["intensity_cumulative_weighted_sum"]),
        "intensity_density": desc(event["intensity_density"]),
    }

    # ----------------------------
    # Dose-response correlations + OLS (event months)
    # ----------------------------
    drivers = [
        ("duration_weighted_sum", "Duration (weighted sum)"),
        ("intensity_cumulative_weighted_sum", "Intensity (cumulative weighted sum)"),
        ("intensity_density", "Intensity density (effective stress)"),
    ]

    rows = []
    y = pd.to_numeric(event["loss_pct"], errors="coerce").to_numpy()

    for xcol, xname in drivers:
        x = pd.to_numeric(event[xcol], errors="coerce").to_numpy()
        sr, sp, pr, pp = _safe_corr(x, y)
        ols = _ols(y, x)
        rows.append({
            "x": xcol,
            "x_name": xname,
            "spearman_r": sr,
            "spearman_p": sp,
            "pearson_r": pr,
            "pearson_p": pp,
            "ols_n": ols.get("n", np.nan),
            "ols_coef": ols.get("coef", np.nan),
            "ols_p": ols.get("p", np.nan),
            "ols_r2": ols.get("r2", np.nan),
            "ols_intercept": ols.get("intercept", np.nan),
        })

        # plot
        figpath = os.path.join(args.outdir, f"scatter_loss_pct_vs_{xcol}.png")
        _make_scatter(
            figpath,
            x,
            y,
            xlabel=xname,
            ylabel="Loss (%) = (factual_pred - cf_pred) / (|factual_pred| + eps) * 100",
            title=f"Event months dose-response: Loss(%) vs {xname}",
        )

    corr_df = pd.DataFrame(rows)
    corr_out = os.path.join(args.outdir, "dose_response_correlations_event_months.csv")
    corr_df.to_csv(corr_out, index=False)

    # compact table for reviewer-style reporting
    sum_rows = []
    for k, v in summary.items():
        sum_rows.append({
            "variable": k,
            "n": v["n"],
            "mean": v["mean"],
            "median": v["median"],
            "p25": v["p25"],
            "p75": v["p75"],
            "p05": v["p05"],
            "p95": v["p95"],
        })
    sum_df = pd.DataFrame(sum_rows)
    sum_out = os.path.join(args.outdir, "dose_response_summary_event_months.csv")
    sum_df.to_csv(sum_out, index=False)

    # ----------------------------
    # Human-readable summary
    # ----------------------------
    txt = []
    txt.append("================================================================================")
    txt.append("Pretrial-aligned metric (Event months only, isMHW=1)")
    txt.append("================================================================================")
    txt.append(f"Inputs:")
    txt.append(f"  factual:        {args.factual}")
    txt.append(f"  counterfactual: {args.counterfactual}")
    txt.append(f"  data:           {args.data}")
    txt.append("")
    txt.append("Metric definition (aligned to 'MHW is a heat hazard'):")
    txt.append("  loss_abs = factual_pred - cf_pred")
    txt.append("    loss_abs > 0  => removing MHW increases GPP (damage case)")
    txt.append("    loss_abs < 0  => removing MHW decreases GPP (apparent gain under MHW)")
    txt.append("")
    txt.append(f"Percent version:")
    txt.append("  loss_pct = loss_abs / (|factual_pred| + eps) * 100")
    txt.append(f"  eps_mode={args.eps_mode}, eps={eps:.6g}")
    txt.append("")
    txt.append(f"Event months n = {len(event):,d}  (out of total {len(merged):,d})")
    txt.append("")
    # direction share
    if len(event) > 0:
        pos_ratio = float((event["loss_abs"] > 0).mean())
        neg_ratio = float((event["loss_abs"] < 0).mean())
        txt.append(f"Direction in event months:")
        txt.append(f"  loss_abs > 0 (damage): {pos_ratio*100:.2f}%")
        txt.append(f"  loss_abs < 0 (gain):   {neg_ratio*100:.2f}%")
        txt.append("")
        txt.append("Key distribution (event months):")
        txt.append(f"  loss_pct median = {event['loss_pct'].median():.4f}%")
        txt.append(f"  loss_pct p25/p75 = {event['loss_pct'].quantile(0.25):.4f}% / {event['loss_pct'].quantile(0.75):.4f}%")
        txt.append("")

    txt.append("Dose-response (event months): correlations + OLS on loss_pct")
    txt.append(corr_df.to_string(index=False))
    txt.append("")
    txt.append("[OUT]")
    txt.append(f"  - {event_out}")
    txt.append(f"  - {sum_out}")
    txt.append(f"  - {corr_out}")
    txt.append(f"  - scatter_loss_pct_vs_*.png")
    txt.append("================================================================================")
    txt.append("DONE")
    txt.append("================================================================================")

    txt_out = os.path.join(args.outdir, "summary_pretrial_alignment.txt")
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write("\n".join(txt))

    print("\n".join(txt))


if __name__ == "__main__":
    main()
