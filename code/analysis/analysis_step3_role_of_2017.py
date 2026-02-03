#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_step3_role_of_2017.py

Step 3：评估 2017 年在累计 MHW-associated GPP impact 中的作用

- 输入：Step 1 生成的 annual_global_mhw_impact_cf_minus_factual.csv
- 输出：
  * 2017 年 impact 占比
  * 去除 2017 年后的累计 impact
  * 2017 vs 其他年份的量级对比
"""

import argparse
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--annual",
        required=True,
        help="annual_global_mhw_impact_cf_minus_factual.csv",
    )
    p.add_argument("--year", type=int, default=2017)
    args = p.parse_args()

    df = pd.read_csv(args.annual)

    if "year" not in df.columns or "impact_year" not in df.columns:
        raise ValueError("annual 表中必须包含 year 与 impact_year")

    if args.year not in df["year"].values:
        raise ValueError(f"未找到年份 {args.year}")

    impact_total = df["impact_year"].sum()
    impact_2017 = df.loc[df["year"] == args.year, "impact_year"].iloc[0]
    impact_others = impact_total - impact_2017

    share_2017 = impact_2017 / impact_total if impact_total != 0 else float("nan")

    # 其他年份的典型量级（绝对值中位数）
    typical_other = (
        df.loc[df["year"] != args.year, "impact_year"]
        .abs()
        .median()
    )

    print("\n" + "=" * 80)
    print("[STEP 3] Role of year", args.year)
    print("=" * 80)
    print(f"Total impact (all years)       = {impact_total:.6e}")
    print(f"Impact in {args.year}              = {impact_2017:.6e}")
    print(f"Impact excluding {args.year}       = {impact_others:.6e}")
    print("-" * 80)
    print(f"Share of {args.year} in total impact = {share_2017:.2%}")
    print(f"Typical |impact_year| (other years) = {typical_other:.6e}")
    print("=" * 80)

    if impact_others < 0:
        print(f"[RESULT] Removing {args.year} does NOT change the sign of cumulative impact.")
    else:
        print(f"[RESULT] Removing {args.year} CHANGES the sign of cumulative impact.")

    print("=" * 80)


if __name__ == "__main__":
    main()
