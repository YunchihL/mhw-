#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_step5q_event_month_spatiotemporal_patterns.py

Step 5q｜MHW 当月（isMHW=1）ΔGPP 的时空特征（含南北半球季节拆分）

定义（落实你的逻辑）：
  delta_gpp = gpp_cf - gpp_factual

解释：
  delta_gpp < 0  → facilitation（促进）：移除 MHW 后 GPP 更低 → MHW 存在时更高
  delta_gpp > 0  → suppression（抑制）：移除 MHW 后 GPP 更高 → MHW 存在时更低
  delta_gpp = 0  → neutral

经纬度列（固定）：
  lat_c, lon_c

输出：
  - step5q_event_rows.csv
  - step5q_by_month.csv / year / latband / lonband / latlonband
  - step5q_by_month_hemisphere.csv（南北半球月份分布）
  - step5q_month_hemisphere_bar.png（两个子图：NH/SH；y = fac% - sup%）
  - step5q_highlights.txt
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


KEY_COLS = ["grid_id", "year", "month"]


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def add_bands(series: pd.Series, step: float, name: str):
    x = pd.to_numeric(series, errors="coerce")
    lo = np.floor(np.nanmin(x) / step) * step
    hi = np.ceil(np.nanmax(x) / step) * step
    if hi <= lo:
        hi = lo + step
    bins = np.arange(lo, hi + step, step)
    labels = [f"{bins[i]:.1f}~{bins[i+1]:.1f}" for i in range(len(bins) - 1)]
    out = pd.cut(x, bins=bins, labels=labels, include_lowest=True)
    out.name = name
    return out.astype("object")


def summarize_group(df: pd.DataFrame, group_col: str):
    """
    统计每个组：
      - facilitation / suppression / neutral 计数
      - 只在非0（fac + sup）里计算占比
      - 产出 fac% - sup% 的差值指标（你要的直观表达）
    """
    rows = []
    for k, sub in df.groupby(group_col, dropna=False):
        fac = (sub["effect"] == "facilitation").sum()  # delta<0
        sup = (sub["effect"] == "suppression").sum()   # delta>0
        neu = (sub["effect"] == "neutral").sum()
        denom = max(1, fac + sup)

        frac_fac = fac / denom
        frac_sup = sup / denom
        score = frac_fac - frac_sup  # 你要求的“<0% - >0%”等价：fac% - sup%

        rows.append({
            group_col: k,
            "n": int(len(sub)),
            "n_facilitation": int(fac),
            "n_suppression": int(sup),
            "n_neutral": int(neu),
            "frac_facilitation_nonzero": float(frac_fac),
            "frac_suppression_nonzero": float(frac_sup),
            "score_fac_minus_sup": float(score),
            "mean_delta": float(sub["delta_gpp"].mean()),
            "median_delta": float(sub["delta_gpp"].median()),
            "p25_delta": float(sub["delta_gpp"].quantile(0.25)),
            "p75_delta": float(sub["delta_gpp"].quantile(0.75)),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("score_fac_minus_sup", ascending=False)
        .reset_index(drop=True)
    )


def top_bottom_k(df, k=5, min_n=50):
    """
    用 score_fac_minus_sup 排序：
      top: facilitation 占主导
      bottom: suppression 占主导
    """
    if df.empty:
        return None, None
    df2 = df[df["n"] >= min_n].copy()
    if df2.empty:
        return None, None
    top_k = df2.sort_values("score_fac_minus_sup", ascending=False).head(k)
    bottom_k = df2.sort_values("score_fac_minus_sup", ascending=True).head(k)
    return top_k, bottom_k


def month_hemisphere_summary(ev: pd.DataFrame, min_n: int):
    """
    输出南北半球分别的月份统计表（并强制 month=1..12 排序）。
    """
    out = []
    for hemi, sub in ev.groupby("hemisphere"):
        tmp = summarize_group(sub, "month")
        tmp["hemisphere"] = hemi
        # 补齐1..12（没有就 NaN / 0）
        all_months = pd.DataFrame({"month": list(range(1, 13))})
        tmp = all_months.merge(tmp, on="month", how="left")
        tmp["hemisphere"] = hemi
        # 对缺失月份填充
        fill0_cols = [
            "n", "n_facilitation", "n_suppression", "n_neutral",
        ]
        for c in fill0_cols:
            tmp[c] = tmp[c].fillna(0).astype(int)
        fillna_cols = [
            "frac_facilitation_nonzero", "frac_suppression_nonzero", "score_fac_minus_sup",
            "mean_delta", "median_delta", "p25_delta", "p75_delta"
        ]
        for c in fillna_cols:
            tmp[c] = tmp[c].fillna(np.nan)
        # 加一个有效性标记：样本量不足
        tmp["is_valid_n"] = tmp["n"] >= min_n
        out.append(tmp)

    return pd.concat(out, ignore_index=True)


def plot_month_bar_two_panels(df_mh: pd.DataFrame, out_png: str, min_n: int):
    """
    画两子图：NH / SH，y=score_fac_minus_sup（fac% - sup%）
    """
    hemis = ["NH", "SH"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

    for ax, hemi in zip(axes, hemis):
        sub = df_mh[df_mh["hemisphere"] == hemi].copy()
        sub = sub.sort_values("month")
        y = sub["score_fac_minus_sup"].values.astype(float)

        # 样本量不足的月份用 0 但用浅色边框提示（这里简单用 hatch）
        valid = sub["is_valid_n"].fillna(False).values

        bars = ax.bar(sub["month"].values, np.nan_to_num(y, nan=0.0))
        for i, b in enumerate(bars):
            if not valid[i]:
                b.set_hatch("//")

        ax.axhline(0, linewidth=1)
        ax.set_title(f"{hemi} | Monthly (fac% - sup%) in MHW months\n(hatched = n < {min_n})")
        ax.set_xlabel("Month")
        ax.set_xticks(range(1, 13))

    axes[0].set_ylabel("facilitation% − suppression% (nonzero Δ only)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--factual", required=True)
    p.add_argument("--counterfactual", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--lat_step", type=float, default=10.0)
    p.add_argument("--lon_step", type=float, default=20.0)
    p.add_argument("--min_n", type=int, default=50)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df_f = pd.read_csv(args.factual)
    df_c = pd.read_csv(args.counterfactual)
    df_d = pd.read_csv(args.data)

    # 你问 month 来源：这里 month 来自三表 merge 的 KEY_COLS（grid_id/year/month）
    # 我们用 factual/counterfactual 的 inner merge 来确定“有哪些月份能对齐”
    # 再把 data.csv 的 isMHW/lat/lon 贴上去（validate one_to_one）
    for col in KEY_COLS:
        if col not in df_f.columns or col not in df_c.columns:
            raise ValueError(f"factual/counterfactual 必须包含列: {KEY_COLS}")

    if "lat_c" not in df_d.columns or "lon_c" not in df_d.columns:
        raise ValueError("data.csv 必须包含 lat_c 和 lon_c")
    if "isMHW" not in df_d.columns:
        raise ValueError("data.csv 必须包含 isMHW")

    df = (
        df_f[KEY_COLS + ["gpp_pred"]]
        .rename(columns={"gpp_pred": "gpp_factual"})
        .merge(
            df_c[KEY_COLS + ["gpp_pred"]].rename(columns={"gpp_pred": "gpp_cf"}),
            on=KEY_COLS,
            how="inner",
            validate="one_to_one",
        )
        .merge(
            df_d[KEY_COLS + ["isMHW", "lat_c", "lon_c"]],
            on=KEY_COLS,
            how="left",
            validate="one_to_one",
        )
    )

    df["delta_gpp"] = df["gpp_cf"] - df["gpp_factual"]

    # ✅落实你的逻辑：delta<0 = facilitation（促进）；delta>0 = suppression（抑制）
    df["effect"] = np.where(
        df["delta_gpp"] < 0, "facilitation",
        np.where(df["delta_gpp"] > 0, "suppression", "neutral")
    )

    ev = df[df["isMHW"] == 1].copy()
    if ev.empty:
        raise RuntimeError("No MHW months found (isMHW==1).")

    # hemisphere
    ev["hemisphere"] = np.where(ev["lat_c"] >= 0, "NH", "SH")

    # bands
    ev["lat_band"] = add_bands(ev["lat_c"], args.lat_step, "lat_band")
    ev["lon_band"] = add_bands(ev["lon_c"], args.lon_step, "lon_band")
    ev["latlon_band"] = ev["lat_band"].astype(str) + " | " + ev["lon_band"].astype(str)

    # outputs tables
    ev.to_csv(os.path.join(args.outdir, "step5q_event_rows.csv"), index=False)

    by_month = summarize_group(ev, "month")
    by_year = summarize_group(ev, "year")
    by_lat = summarize_group(ev, "lat_band")
    by_lon = summarize_group(ev, "lon_band")
    by_latlon = summarize_group(ev, "latlon_band")

    by_month.to_csv(os.path.join(args.outdir, "step5q_by_month.csv"), index=False)
    by_year.to_csv(os.path.join(args.outdir, "step5q_by_year.csv"), index=False)
    by_lat.to_csv(os.path.join(args.outdir, "step5q_by_latband.csv"), index=False)
    by_lon.to_csv(os.path.join(args.outdir, "step5q_by_lonband.csv"), index=False)
    by_latlon.to_csv(os.path.join(args.outdir, "step5q_by_latlonband.csv"), index=False)

    # hemisphere-month
    by_month_hemi = month_hemisphere_summary(ev, min_n=args.min_n)
    by_month_hemi.to_csv(os.path.join(args.outdir, "step5q_by_month_hemisphere.csv"), index=False)

    # plot
    out_png = os.path.join(args.outdir, "step5q_month_hemisphere_bar.png")
    plot_month_bar_two_panels(by_month_hemi, out_png, min_n=args.min_n)

    # ------------------------------------------------------------
    # highlight.txt
    # ------------------------------------------------------------
    hl = []
    hl.append("=" * 80)
    hl.append("Step 5q | Highlights: ΔGPP patterns in MHW months (delta = cf - factual)")
    hl.append("=" * 80)
    hl.append("")
    hl.append("Definition (IMPORTANT):")
    hl.append("  delta_gpp = gpp_cf - gpp_factual")
    hl.append("  delta_gpp < 0 : facilitation (MHW promotes contemporaneous GPP)")
    hl.append("  delta_gpp > 0 : suppression  (MHW suppresses contemporaneous GPP)")
    hl.append("")
    hl.append(f"Total MHW event months: {len(ev)}")
    hl.append(f"min_n per group: {args.min_n}")
    hl.append("")

    top_m, bot_m = top_bottom_k(by_month, min_n=args.min_n)
    hl.append("[Seasonality | Month] (ranked by score = fac% - sup%)")
    hl.append("Most facilitation-dominant months:")
    hl.append(top_m.to_string(index=False) if top_m is not None else "NA")
    hl.append("")
    hl.append("Most suppression-dominant months:")
    hl.append(bot_m.to_string(index=False) if bot_m is not None else "NA")
    hl.append("")

    top_lat, bot_lat = top_bottom_k(by_lat, min_n=args.min_n)
    hl.append("[Spatial | Latitude bands] (ranked by score = fac% - sup%)")
    hl.append("Most facilitation-dominant bands:")
    hl.append(top_lat.to_string(index=False) if top_lat is not None else "NA")
    hl.append("")
    hl.append("Most suppression-dominant bands:")
    hl.append(bot_lat.to_string(index=False) if bot_lat is not None else "NA")
    hl.append("")

    top_ll, bot_ll = top_bottom_k(by_latlon, min_n=args.min_n)
    hl.append("[Spatial | Lat × Lon bands] (ranked by score = fac% - sup%)")
    hl.append("Most facilitation-dominant bands:")
    hl.append(top_ll.to_string(index=False) if top_ll is not None else "NA")
    hl.append("")
    hl.append("Most suppression-dominant bands:")
    hl.append(bot_ll.to_string(index=False) if bot_ll is not None else "NA")
    hl.append("")
    hl.append("=" * 80)

    with open(os.path.join(args.outdir, "step5q_highlights.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(hl))

    print("=" * 80)
    print("[STEP 5q OK] Spatiotemporal patterns + hemisphere monthly bar generated")
    print("[OUT]")
    print(" - step5q_event_rows.csv")
    print(" - step5q_by_month.csv")
    print(" - step5q_by_latband.csv")
    print(" - step5q_by_latlonband.csv")
    print(" - step5q_by_month_hemisphere.csv")
    print(" - step5q_month_hemisphere_bar.png")
    print(" - step5q_highlights.txt")
    print("=" * 80)


if __name__ == "__main__":
    main()
