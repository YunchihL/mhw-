
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_summarize_grid_contributions.py

作用：
- 输出你要的三条信息（反事实年尺度 → grid 分解）：
  1) n_year_neg >= n_year_pos 的 grid 占比
  2) Top 10 负贡献 grid 的多年平均 ΔGPP_year 之和，占“总负贡献”的百分比
  3) 2017 年是“很多 grid 变正”还是“少数 grid 极端变正”
- 优先读取 grid_annual_delta_gpp.csv / grid_multiyear_summary.csv
  若不存在，则从 factual/counterfactual rolling predictions 现场计算。

用法（推荐：读取已生成文件）：
python code/analysis/analysis_summarize_grid_contributions.py \
  --outdir code/analysis/results

用法（若你还没生成 grid_annual_delta_gpp.csv 等文件）：
python code/analysis/analysis_summarize_grid_contributions.py \
  --factual results/predictions/factual_rolling_predictions.csv \
  --counterfactual results/predictions/counterfactual_rolling_predictions.csv \
  --outdir code/analysis/results
"""

import argparse
import os
import numpy as np
import pandas as pd


DEFAULT_OUTDIR = "code/analysis/results"
DEFAULT_GRID_YEAR = os.path.join(DEFAULT_OUTDIR, "grid_annual_delta_gpp.csv")
DEFAULT_SUMMARY = os.path.join(DEFAULT_OUTDIR, "grid_multiyear_summary.csv")


def _load_or_build_grid_year_and_summary(
    outdir: str,
    factual: str | None,
    counterfactual: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    优先加载：
      - grid_annual_delta_gpp.csv
      - grid_multiyear_summary.csv
    否则从预测文件构建（并写回 outdir）。
    """
    grid_year_path = os.path.join(outdir, "grid_annual_delta_gpp.csv")
    summary_path = os.path.join(outdir, "grid_multiyear_summary.csv")

    if os.path.exists(grid_year_path) and os.path.exists(summary_path):
        grid_year = pd.read_csv(grid_year_path)
        summary = pd.read_csv(summary_path)
        return grid_year, summary

    if not factual or not counterfactual:
        raise FileNotFoundError(
            "未找到 grid_annual_delta_gpp.csv / grid_multiyear_summary.csv，且未提供 --factual/--counterfactual 用于现场计算。"
        )

    df_f = pd.read_csv(factual)
    df_c = pd.read_csv(counterfactual)

    key_cols = ["grid_id", "year", "month"]
    for c in key_cols:
        if c not in df_f.columns or c not in df_c.columns:
            raise ValueError(f"缺少必要列: {c}")
    if "gpp_pred" not in df_f.columns or "gpp_pred" not in df_c.columns:
        raise ValueError("缺少 gpp_pred 列")

    df = (
        df_f[key_cols + ["gpp_pred"]]
        .rename(columns={"gpp_pred": "gpp_factual"})
        .merge(
            df_c[key_cols + ["gpp_pred"]].rename(columns={"gpp_pred": "gpp_cf"}),
            on=key_cols,
            how="inner",
        )
    )
    if len(df) == 0:
        raise RuntimeError("合并后数据为空，请检查两份文件的 (grid_id, year, month) 是否一致")

    df["delta_gpp_month"] = df["gpp_cf"] - df["gpp_factual"]

    grid_year = (
        df.groupby(["grid_id", "year"], as_index=False)
        .agg(delta_gpp_year=("delta_gpp_month", "sum"),
             n_months=("delta_gpp_month", "count"))
    )

    summary = (
        grid_year.groupby("grid_id", as_index=False)
        .agg(
            n_years=("year", "count"),
            n_year_pos=("delta_gpp_year", lambda x: (x > 0).sum()),
            n_year_neg=("delta_gpp_year", lambda x: (x < 0).sum()),
            mean_delta_gpp_year=("delta_gpp_year", "mean"),
            median_delta_gpp_year=("delta_gpp_year", "median"),
        )
    )
    summary["pos_ratio"] = summary["n_year_pos"] / summary["n_years"]

    os.makedirs(outdir, exist_ok=True)
    grid_year.to_csv(grid_year_path, index=False)
    summary.to_csv(summary_path, index=False)

    return grid_year, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help="输出目录（建议 code/analysis/results）")
    parser.add_argument("--grid-year", type=str, default=None, help="grid_annual_delta_gpp.csv 路径（可选）")
    parser.add_argument("--summary", type=str, default=None, help="grid_multiyear_summary.csv 路径（可选）")
    parser.add_argument("--factual", type=str, default=None, help="factual rolling predictions（若需现场计算）")
    parser.add_argument("--counterfactual", type=str, default=None, help="counterfactual rolling predictions（若需现场计算）")
    parser.add_argument("--year_focus", type=int, default=2017, help="重点年份（默认 2017）")
    parser.add_argument("--topk", type=int, default=10, help="Top K 负贡献 grid（默认 10）")
    args = parser.parse_args()

    outdir = args.outdir

    # 如果用户明确给了路径，就按给的路径读；否则走 load_or_build
    if args.grid_year and args.summary:
        grid_year = pd.read_csv(args.grid_year)
        summary = pd.read_csv(args.summary)
    else:
        grid_year, summary = _load_or_build_grid_year_and_summary(outdir, args.factual, args.counterfactual)

    # -----------------------------
    # 信息 1：n_year_neg >= n_year_pos 的 grid 占比
    # -----------------------------
    cond = summary["n_year_neg"] >= summary["n_year_pos"]
    ratio = cond.mean() * 100
    n_grids = len(summary)
    n_cond = int(cond.sum())

    # -----------------------------
    # 信息 2：Top K 负贡献 grid 的贡献占比
    # 这里用多年平均的 mean_delta_gpp_year 来排序（越小越负）
    # “总负贡献”定义为：所有 grid 的负 mean_delta_gpp_year 之和（取绝对值做分母）
    # -----------------------------
    topk = int(args.topk)
    summary_sorted = summary.sort_values("mean_delta_gpp_year")  # 最负在前
    topk_df = summary_sorted.head(topk).copy()

    total_neg = summary.loc[summary["mean_delta_gpp_year"] < 0, "mean_delta_gpp_year"].sum()  # 负数
    topk_neg = topk_df.loc[topk_df["mean_delta_gpp_year"] < 0, "mean_delta_gpp_year"].sum()  # 负数

    # 防止没有负值的极端情况
    if total_neg == 0:
        topk_share = np.nan
    else:
        topk_share = (abs(topk_neg) / abs(total_neg)) * 100

    # -----------------------------
    # 信息 3：year_focus（默认2017） 的翻正结构
    # 定义：
    # - 在该年 grid 的 delta_gpp_year > 0 的比例
    # - “贡献集中度”：Top 5 正贡献 grid 占该年总正贡献的比例
    # -----------------------------
    yf = int(args.year_focus)
    gy = grid_year[grid_year["year"] == yf].copy()
    if len(gy) == 0:
        raise ValueError(f"在 grid_annual_delta_gpp 中找不到 year={yf} 的记录，请确认年份覆盖。")

    pos = gy["delta_gpp_year"] > 0
    pos_ratio = pos.mean() * 100
    n_pos = int(pos.sum())
    n_total_y = len(gy)

    pos_sum = gy.loc[pos, "delta_gpp_year"].sum()
    gy_pos_sorted = gy.loc[pos].sort_values("delta_gpp_year", ascending=False)

    top5_pos = gy_pos_sorted.head(5)["delta_gpp_year"].sum()
    conc = (top5_pos / pos_sum) * 100 if pos_sum > 0 else np.nan

    # -----------------------------
    # 输出到控制台
    # -----------------------------
    print("\n" + "=" * 80)
    print("[1] Grid 方向占比（多年）")
    print("=" * 80)
    print(f"n_grids = {n_grids}")
    print(f"n_year_neg >= n_year_pos : {n_cond} / {n_grids}  ({ratio:.2f}%)")

    print("\n" + "=" * 80)
    print(f"[2] Top {topk} 负贡献 grid 的占比（多年平均ΔGPP_year）")
    print("=" * 80)
    print(f"total_neg (sum of negative mean ΔGPP_year) = {total_neg:.6g}")
    print(f"top{topk}_neg (sum) = {topk_neg:.6g}")
    print(f"Top {topk} share of total negative contribution = {topk_share:.2f}%")
    print("\nTop grids preview:")
    print(topk_df[["grid_id", "n_years", "n_year_pos", "n_year_neg", "mean_delta_gpp_year", "pos_ratio"]].to_string(index=False))

    print("\n" + "=" * 80)
    print(f"[3] {yf} 年翻正结构（grid 层面）")
    print("=" * 80)
    print(f"{yf}: positive grids = {n_pos} / {n_total_y}  ({pos_ratio:.2f}%)")
    print(f"{yf}: concentration (Top 5 positive share of total positive) = {conc:.2f}%")
    if len(gy_pos_sorted) > 0:
        print("\nTop 10 positive grids in that year:")
        print(gy_pos_sorted.head(10)[["grid_id", "delta_gpp_year", "n_months"]].to_string(index=False))
    else:
        print("\nNo positive grids in that year.")

    # -----------------------------
    # 导出一份小结果表（便于你写文字时引用）
    # -----------------------------
    os.makedirs(outdir, exist_ok=True)

    # 导出：topk 负贡献 grid（多年）
    topk_out = os.path.join(outdir, f"summary_top{topk}_negative_grids.csv")
    topk_df.to_csv(topk_out, index=False)

    # 导出：year_focus 的 grid 年效应（便于你后续画图/查 grid）
    y_out = os.path.join(outdir, f"grid_delta_gpp_year_{yf}.csv")
    gy.sort_values("delta_gpp_year", ascending=False).to_csv(y_out, index=False)

    # 写一个简短的 txt 摘要（方便你复制到笔记）
    txt_out = os.path.join(outdir, "summary_three_key_findings.txt")
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write(f"[1] n_year_neg >= n_year_pos : {n_cond}/{n_grids} ({ratio:.2f}%)\n")
        f.write(f"[2] Top {topk} share of total negative contribution (multi-year mean): {topk_share:.2f}%\n")
        f.write(f"[3] {yf} positive grids: {n_pos}/{n_total_y} ({pos_ratio:.2f}%), top5 positive concentration: {conc:.2f}%\n")

    print("\n[OUT] 写出文件：")
    print(" -", topk_out)
    print(" -", y_out)
    print(" -", txt_out)
    print("\n[OK] 完成。\n")


if __name__ == "__main__":
    main()
