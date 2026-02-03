#!/usr/bin/env python
"""
将data.csv中的isMHW列从布尔值转换为整数0/1
True -> 1, False -> 0
"""

import pandas as pd
import numpy as np

def main():
    # 读取原始数据
    input_file = "data/data.csv"
    output_file = "data/data.csv"

    print(f"[INFO] 读取数据: {input_file}")
    df = pd.read_csv(input_file)

    # 检查isMHW列是否存在
    if "isMHW" not in df.columns:
        print("错误: 数据中没有isMHW列")
        return

    # 检查当前类型
    print(f"[INFO] isMHW列前10个值: {df['isMHW'].head(10).tolist()}")
    print(f"[INFO] isMHW列数据类型: {df['isMHW'].dtype}")

    # 统计True/False数量
    true_count = (df['isMHW'] == True).sum() if True in df['isMHW'].values else 0
    false_count = (df['isMHW'] == False).sum() if False in df['isMHW'].values else 0
    true_str_count = (df['isMHW'] == 'True').sum() if 'True' in df['isMHW'].values else 0
    false_str_count = (df['isMHW'] == 'False').sum() if 'False' in df['isMHW'].values else 0

    print(f"[INFO] True值数量: {true_count}")
    print(f"[INFO] False值数量: {false_count}")
    print(f"[INFO] 'True'字符串数量: {true_str_count}")
    print(f"[INFO] 'False'字符串数量: {false_str_count}")

    # 转换函数
    def convert_value(x):
        if isinstance(x, bool):
            return 1 if x else 0
        elif isinstance(x, str):
            if x.lower() == 'true':
                return 1
            elif x.lower() == 'false':
                return 0
            else:
                try:
                    # 尝试转换为数字
                    return int(float(x))
                except:
                    return 0
        elif isinstance(x, (int, float, np.integer)):
            return int(x)
        else:
            return 0

    # 应用转换
    df['isMHW'] = df['isMHW'].apply(convert_value)

    # 验证转换结果
    print(f"[INFO] 转换后isMHW列前10个值: {df['isMHW'].head(10).tolist()}")
    print(f"[INFO] 转换后数据类型: {df['isMHW'].dtype}")
    print(f"[INFO] 转换后统计 - 1 (True): {(df['isMHW'] == 1).sum()}")
    print(f"[INFO] 转换后统计 - 0 (False): {(df['isMHW'] == 0).sum()}")

    # 保存回原文件
    print(f"[INFO] 保存到: {output_file}")
    df.to_csv(output_file, index=False)

    print("[INFO] 转换完成!")

    # 同时更新反事实数据文件（如果存在）
    cf_file = "data/data_counterfactual.csv"
    try:
        if pd.io.common.file_exists(cf_file):
            print(f"[INFO] 同时更新反事实数据: {cf_file}")
            cf_df = pd.read_csv(cf_file)
            if "isMHW" in cf_df.columns:
                cf_df['isMHW'] = 0  # 反事实数据中isMHW应为0
                cf_df.to_csv(cf_file, index=False)
                print(f"[INFO] 反事实数据已更新")
    except:
        print(f"[INFO] 跳过反事实数据更新（文件可能不存在）")

if __name__ == "__main__":
    main()