#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_check_predictions_sanity.py

作用：
- 对预测结果（factual/counterfactual rolling predictions）做“数值域合理性”体检（1.2）
- 重点检查：负值、零值、极端值、量级、以及按 year/month/grid 的集中分布
- 输出：控制台报告 + 可选导出 csv（异常样本行）

用法示例：
1) 自动使用默认路径（若存在）：
   python analysis_check_predictions_sanity.py

2) 手动指定文件：
   python analysis_check_predictions_sanity.py \
     --factual results/predictions/factual_rolling_predictions.csv \
     --counterfactual results/predictions/counterfactual_rolling_predictions.csv

3) 只检查一个文件：
   python analysis_check_predictions_sanity.py --factual path/to/file.csv

可选：
   --outdir /abs/path/to/results/tables
"""

from __future__ import annotations

import os
import argparse
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_FACTUAL = "/home/linyunzhi/project_for_journal/results/predictions/factual_rolling_predictions.csv"
DEFAULT_COUNTERFACTUAL = "/home/linyunzhi/project_for_journal/results/predictions/counterfactual_rolling_predictions.csv"


def _pick_first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _infer_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    尝试推断 gpp_true / gpp_pred 列名（兼容可能的命名）
    """
    true_candidates = ["gpp_true", "y_true", "true", "gpp_obs", "observed", "gpp_total_true"]
    pred_candidates = ["gpp_pred", "y_pred", "pred", "prediction", "gpp_total_pred"]

    true_col = next((c for c in true_candidates if c in df.columns), None)
    pred_col = next((c for c in pred_candidates if c in df.columns), None)

    # 模糊匹配（包含 gpp + true/pred）
    if true_col is None:
        for c in df.columns:
            cl = c.lower()
            if "gpp" in cl and "true" in cl:
                true_col = c
                break
    if pred_col is None:
        for c in df.columns:
            cl = c.lower()
            if "gpp" in cl and ("pred" in cl or "prediction" in cl):
                pred_col = c
                break

    return true_col, pred_col


def _ensure_year_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    确保存在 year/month；若存在 date/time 类列则尝试解析生成。
    """
    if "year" in df.columns and "month" in df.columns:
        return df

    # 常见日期列名
    date_candidates = ["date", "time", "timestamp", "datetime", "ym", "year_month"]
    date_col = next((c for c in date_candidates if c in df.columns), None)

    if date_col is not None:
        dt = pd.to_datetime(df[date_col], errors="coerce")
        if dt.notna().any():
            df = df.copy()
            df["year"] = dt.dt.year
            df["month"] = dt.dt.month
            return df

    return df


def _basic_stats(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce")
    if not s.notna().any():
        return {
            "count": 0, "min": np.nan, "p01": np.nan, "p05": np.nan, "p50": np.nan,
            "p95": np.nan, "p99": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan
        }
    v = s.values.astype(float)
    return {
        "count": int(np.isfinite(v).sum()),
        "min": float(np.nanmin(v)),
        "p01": float(np.nanpercentile(v, 1)),
        "p05": float(np.nanpercentile(v, 5)),
        "p50": float(np.nanpercentile(v, 50)),
        "p95": float(np.nanpercentile(v, 95)),
        "p99": float(np.nanpercentile(v, 99)),
        "max": float(np.nanmax(v)),
        "mean": float(np.nanmean(v)),
        "std": float(np.nanstd(v)),
    }


def _print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def analyze_one_file(path: str, outdir: Optional[str] = None) -> None:
    _print_header(f"[CHECK] {path}")
    df = pd.read_csv(path)
    df = _ensure_year_month(df)

    true_col, pred_col = _infer_columns(df)
    if true_col is None or pred_col is None:
        print("[ERROR] 无法自动识别 gpp_true / gpp_pred 列名。")
        print("  现有列名：", list(df.columns))
        print("  建议：将列名改为 gpp_true / gpp_pred，或在脚本里补充候选名。")
        return

    # 转数值
    y_true = pd.to_numeric(df[true_col], errors="coerce")
    y_pred = pd.to_numeric(df[pred_col], errors="coerce")

    print(f"[INFO] 使用列名：true='{true_col}', pred='{pred_col}'")
    print(f"[INFO] 样本数：{len(df):,} | true缺失：{int(y_true.isna().sum()):,} | pred缺失：{int(y_pred.isna().sum()):,}")

    # 1) 基本统计
    _print_header("1) 基本统计（true / pred）")
    print("TRUE:", _basic_stats(y_true))
    print("PRED:", _basic_stats(y_pred))

    # 2) 负值/零值
    _print_header("2) 负值 / 零值检查")
    n_pred_neg = int((y_pred < 0).sum())
    n_true_neg = int((y_true < 0).sum())
    n_pred_zero = int((y_pred == 0).sum())
    n_true_zero = int((y_true == 0).sum())

    print(f"pred < 0 : {n_pred_neg:,} ({n_pred_neg/len(df)*100:.2f}%)")
    print(f"pred = 0 : {n_pred_zero:,} ({n_pred_zero/len(df)*100:.2f}%)")
    print(f"true < 0 : {n_true_neg:,} ({n_true_neg/len(df)*100:.2f}%)")
    print(f"true = 0 : {n_true_zero:,} ({n_true_zero/len(df)*100:.2f}%)")

    # 3) 误差与相对误差
    _print_header("3) 误差体检（abs / relative, with epsilon）")
    err = y_pred - y_true

    true_pos = y_true[y_true > 0]
    eps = float(np.nanpercentile(true_pos.values, 5)) if len(true_pos) else 1e-6

    rel = err / (y_true.replace(0, np.nan))
    rel_eps = err / (y_true + eps)

    print(f"[INFO] epsilon (true_pos p05) = {eps:.6g}")
    print("[ABS_ERR] ", _basic_stats(err))
    print("[REL_ERR  (err/true, zero->nan)] ", _basic_stats(rel))
    print("[REL_ERR_EPS (err/(true+eps))] ", _basic_stats(rel_eps))

    extreme = (rel_eps.abs() > 10)  # ~>1000%
    print(f"|rel_err_eps| > 10 (≈1000%): {int(extreme.sum()):,} ({extreme.mean()*100:.2f}%)")

    # 4) 异常集中位置
    _print_header("4) 异常集中位置（year / month / grid）")

    has_year_month = ("year" in df.columns and "month" in df.columns)
    has_grid = ("grid_id" in df.columns)

    # 标记异常：pred<0 或 true==0 或 |rel_err_eps|>10
    flag = (y_pred < 0) | (y_true == 0) | extreme
    print(f"[INFO] 异常标记总数: {int(flag.sum()):,} / {len(df):,} ({flag.mean()*100:.2f}%)")

    if has_year_month:
        by_ym = (
            df.assign(_flag=flag)
              .groupby(["year", "month"])["_flag"]
              .agg(n_flag="sum", flag_rate="mean")
              .reset_index()
              .sort_values(["flag_rate", "n_flag"], ascending=False)
              .head(12)
        )
        print("\n[TOP 12] flag_rate highest by (year, month):")
        print(by_ym.to_string(index=False))
    else:
        print("[WARN] 未发现 year/month 列，跳过 year-month 定位。")

    if has_grid:
        by_grid = (
            df.assign(_flag=flag, _abs_err=err.abs())
              .groupby("grid_id", as_index=False)
              .agg(
                  n=("grid_id", "size"),
                  flag_rate=("_flag", "mean"),
                  n_flag=("_flag", "sum"),
                  med_abs_err=("_abs_err", "median"),
                  med_true=(true_col, "median"),
                  med_pred=(pred_col, "median"),
              )
              .sort_values(["flag_rate", "med_abs_err"], ascending=False)
              .head(15)
        )
        print("\n[TOP 15] problematic grids (flag_rate, median abs err):")
        print(by_grid.to_string(index=False))
    else:
        print("[WARN] 未发现 grid_id 列，跳过 grid 定位。")

    # 5) 可选导出异常样本
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        base = os.path.splitext(os.path.basename(path))[0]

        # 尽量保留可追溯列
        keep_pref = ["grid_id", "pos_id", "lat", "lon", "year", "month", "time", "date"]
        cols_keep = [c for c in keep_pref if c in df.columns]
        cols_keep += [true_col, pred_col]
        # 再补充一些常见关键字段（如果存在）
        for extra in ["isMHW", "mhw_intensity", "mhw_cumint", "mhw_duration", "mean_sst"]:
            if extra in df.columns and extra not in cols_keep:
                cols_keep.append(extra)

        out_flag = df.loc[flag, cols_keep].copy()
        out_path = os.path.join(outdir, f"{base}__flagged_rows.csv")
        out_flag.to_csv(out_path, index=False)
        print(f"\n[OUT] 异常样本已导出: {out_path}")

    print("\n[OK] 该文件检查完成。\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factual", type=str, default=None, help="factual rolling predictions csv")
    parser.add_argument("--counterfactual", type=str, default=None, help="counterfactual rolling predictions csv")
    parser.add_argument("--outdir", type=str, default=None, help="optional output dir to save flagged rows")
    args = parser.parse_args()

    factual = args.factual or _pick_first_existing([DEFAULT_FACTUAL])
    counterfactual = args.counterfactual or _pick_first_existing([DEFAULT_COUNTERFACTUAL])

    if factual is None and counterfactual is None:
        print("[ERROR] 找不到默认预测文件。请用 --factual / --counterfactual 指定路径。")
        print(f"  默认尝试: {DEFAULT_FACTUAL}")
        print(f"          : {DEFAULT_COUNTERFACTUAL}")
        return

    if factual:
        analyze_one_file(factual, outdir=args.outdir)
    if counterfactual:
        analyze_one_file(counterfactual, outdir=args.outdir)


if __name__ == "__main__":
    main()
