#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_compare_grid_groups.py

作用：
- 将 grid 按“多年 ΔGPP 方向特征”分组：
    NEG: n_year_neg > n_year_pos
    OTHERS: 其余
- 对比两组 grid 的空间 / 生态背景特征（描述统计）
"""

import argparse
import os
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        required=True,
        help="grid_multiyear_summary.csv 路径",
    )
    parser.add_argument(
        "--factual",
        required=True,
        help="factual rolling predictions csv（用于提取背景变量）",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="输出 csv 路径",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------
    # 1. 读取数据
    # ------------------------------------------------------------
    summary = pd.read_csv(args.summary)
    df = pd.read_csv(args.factual)

    if "grid_id" not in summary.columns:
        raise ValueError("summary 缺少 grid_id")
    if "grid_id" not in df.columns:
        raise ValueError("factual 缺少 grid_id")

    # ------------------------------------------------------------
    # 2. 定义分组
    # ------------------------------------------------------------
    summary["group"] = np.where(
        summary["n_year_neg"] > summary["n_year_pos"],
        "NEG",
        "OTHERS",
    )

    grid_group = summary[["grid_id", "group"]]

    # ------------------------------------------------------------
    # 3. 提取 grid 层面的背景特征
    #    （对 factual 在 grid × time 上求多年平均）
    # ------------------------------------------------------------
    candidate_vars = [
        "lat", "lon",
        "gpp_true", "gpp_pred",
        "mean_sst",
        "isMHW",
    ]
    candidate_vars = [v for v in candidate_vars if v in df.columns]

    if len(candidate_vars) == 0:
        raise RuntimeError("factual 中未找到可用的背景变量")

    grid_bg = (
        df.groupby("grid_id")[candidate_vars]
        .mean()
        .reset_index()
    )

    # isMHW 若是 0/1，mean 就是“年内发生比例”

    # ------------------------------------------------------------
    # 4. 合并分组信息
    # ------------------------------------------------------------
    grid_bg = grid_bg.merge(grid_group, on="grid_id", how="inner")

    # ------------------------------------------------------------
    # 5. 组间对比（描述统计）
    # ------------------------------------------------------------
    rows = []
    for var in candidate_vars:
        for grp in ["NEG", "OTHERS"]:
            x = grid_bg.loc[grid_bg["group"] == grp, var].dropna()
            rows.append({
                "variable": var,
                "group": grp,
                "n_grids": len(x),
                "mean": x.mean(),
                "median": x.median(),
                "p25": x.quantile(0.25),
                "p75": x.quantile(0.75),
            })

    out_df = pd.DataFrame(rows)

    # ------------------------------------------------------------
    # 6. 输出
    # ------------------------------------------------------------
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)

    print("\n[OK] Grid 分组对比完成")
    print(out_df)


if __name__ == "__main__":
    main()
