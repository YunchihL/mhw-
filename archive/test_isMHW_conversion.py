#!/usr/bin/env python
"""
测试 isMHW 转换是否正常工作
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from code.utils.data_utils import preprocess_data

def test_data_csv():
    print("=== 测试 data.csv ===")
    df = pd.read_csv("data/data.csv")
    print(f"数据形状: {df.shape}")
    print(f"isMHW 列类型: {df['isMHW'].dtype}")
    print(f"isMHW 唯一值: {df['isMHW'].unique()}")
    print(f"isMHW 值统计:")
    print(df['isMHW'].value_counts())

    # 应用预处理
    config = {"data": {"path": "data/data.csv"}}
    df_proc, scaler = preprocess_data(df, config)

    print(f"\n预处理后 isMHW 列类型: {df_proc['isMHW'].dtype}")
    print(f"预处理后 isMHW 唯一值: {df_proc['isMHW'].unique()}")
    print(f"预处理后 isMHW 值统计:")
    print(df_proc['isMHW'].value_counts())

    # 验证转换是否正确
    expected_values = set(["False", "True"])
    actual_values = set(df_proc['isMHW'].unique())
    if actual_values == expected_values:
        print("✓ isMHW 转换正确: 值为 'False' 和 'True'")
    else:
        print(f"✗ isMHW 转换错误: 期望 {expected_values}, 实际 {actual_values}")
        return False

    return True

def test_counterfactual_csv():
    print("\n=== 测试 data_counterfactual.csv ===")
    cf_path = "data/data_counterfactual.csv"
    if not os.path.exists(cf_path):
        print(f"文件不存在: {cf_path}")
        return True

    df = pd.read_csv(cf_path)
    print(f"数据形状: {df.shape}")
    print(f"isMHW 列类型: {df['isMHW'].dtype}")
    print(f"isMHW 唯一值: {df['isMHW'].unique()}")
    print(f"isMHW 值统计:")
    print(df['isMHW'].value_counts())

    # 应用预处理
    config = {"data": {"path": cf_path}}
    df_proc, scaler = preprocess_data(df, config)

    print(f"\n预处理后 isMHW 列类型: {df_proc['isMHW'].dtype}")
    print(f"预处理后 isMHW 唯一值: {df_proc['isMHW'].unique()}")
    print(f"预处理后 isMHW 值统计:")
    print(df_proc['isMHW'].value_counts())

    # 反事实数据中 isMHW 应该全部为 "False"
    if set(df_proc['isMHW'].unique()) == set(["False"]):
        print("✓ 反事实数据 isMHW 转换正确: 全部为 'False'")
    else:
        print(f"✗ 反事实数据 isMHW 转换错误: 期望全部为 'False', 实际 {set(df_proc['isMHW'].unique())}")
        return False

    return True

def main():
    print("测试 isMHW 数据类型转换")
    print("="*50)

    success = True
    if not test_data_csv():
        success = False

    if not test_counterfactual_csv():
        success = False

    print("\n" + "="*50)
    if success:
        print("✓ 所有测试通过")
    else:
        print("✗ 测试失败")
        sys.exit(1)

if __name__ == "__main__":
    main()