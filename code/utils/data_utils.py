# ============================================================
#  data_utils.py
#  数据加载与预处理（最终结构）
# ============================================================

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# 1. 加载 CSV
# ------------------------------------------------------------
def load_data(config):
    data_path = config["data"]["path"]
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"[INFO] Loaded data shape: {df.shape}")
    return df


# ------------------------------------------------------------
# 2. 预处理：特征工程 + 标准化
# ------------------------------------------------------------
def preprocess_data(df, config):

    # === 月份编码 ===
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # === 填补 MHW 特征缺失 ===
    if (
        "intensity_density" not in df.columns
        and "intensity_cumulative_weighted_sum" in df.columns
        and "duration_weighted_sum" in df.columns
    ):
        df["intensity_density"] = np.where(
            df["duration_weighted_sum"] > 0,
            df["intensity_cumulative_weighted_sum"] / df["duration_weighted_sum"],
            0.0,
        )

    mhw_cols = [
        "intensity_max_month",
        "duration_weighted_sum",
        "intensity_cumulative_weighted_sum",
        "intensity_density",
    ]
    for c in mhw_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # === 按 grid_id 进行排序 ===
    df = df.sort_values(by=["grid_id", "year", "month"]).reset_index(drop=True)

    # === 标准化目标 gpp_total ===
    scaler = StandardScaler()
    df["gpp_total_norm"] = scaler.fit_transform(df[["gpp_total"]])

    # 模型只读 gpp_total_norm，因此覆盖旧列
    df["gpp_total"] = df["gpp_total_norm"]
    df.drop(columns=["gpp_total_norm"], inplace=True)

    return df, scaler
