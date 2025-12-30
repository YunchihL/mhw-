# ============================================================
# factual_predict.py
#
# 功能：
# - 使用训练好的 TFT 模型
# - 对所有 grid、所有可预测时间点做 factual 预测
# - 输出表包含：
#     grid_id / year / month / gpp_true / gpp_pred
#
# 设计原则：
# - 只使用 model.predict（不手写 forward）
# - 只使用 decoder_time_idx 对齐原始 df
# - 不使用 validation.index
# ============================================================

import os
import argparse
import numpy as np
import pandas as pd

from pytorch_forecasting.models import TemporalFusionTransformer
from code.train.train_tft import (
    get_config,
    get_raw_data,
    preprocess,
    create_datasets,
)


# ------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------
def get_project_root():
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )


def resolve_ckpt_path(ckpt_arg):
    project_root = get_project_root()
    ckpt_dir = os.path.join(project_root, "checkpoints")

    if ckpt_arg is None:
        ckpts = sorted(
            [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")],
            reverse=True,
        )
        if not ckpts:
            raise FileNotFoundError("❌ checkpoints/ 中没有 ckpt")
        print(f"[INFO] 使用最新 ckpt: {ckpts[0]}")
        return os.path.join(ckpt_dir, ckpts[0])

    if os.path.exists(ckpt_arg):
        return ckpt_arg

    candidate = os.path.join(ckpt_dir, ckpt_arg)
    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError(f"❌ 找不到 ckpt: {ckpt_arg}")


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()

    project_root = get_project_root()

    # === 1. 加载模型 ===
    ckpt_path = resolve_ckpt_path(args.ckpt)
    model = TemporalFusionTransformer.load_from_checkpoint(
        ckpt_path, strict=False
    )
    model.eval()
    print(f"[INFO] Loaded model from {ckpt_path}")

    # === 2. 加载 & 预处理数据 ===
    config = get_config()
    df_raw = get_raw_data(config)
    df_proc, scaler = preprocess(df_raw, config)

    # === 3. 构建 dataset / dataloader ===
    _, validation = create_datasets(df_proc, config)
    val_dl = validation.to_dataloader(
        train=False, batch_size=64, num_workers=0
    )

    # === 4. 统一预测入口（关键） ===
    pred_out = model.predict(
        val_dl,
        return_x=True,
        mode="prediction",
    )

    # pred_out[0]: predictions (normalized)
    # pred_out[1]: x
    # === model.predict 输出 ===
    preds_norm = pred_out[0].detach().cpu().numpy()

    x = pred_out[1]

# === decoder target（真实值）===
    y_true_norm = x["decoder_target"].detach().cpu().numpy()

# === decoder 时间索引 ===
    time_idx = x["decoder_time_idx"].detach().cpu().numpy()

# ------------------------------------------------------------
# 统一展开（保证长度完全一致）
# ------------------------------------------------------------
    preds_norm = preds_norm.reshape(-1)
    y_true_norm = y_true_norm.reshape(-1)
    time_idx = time_idx.reshape(-1).astype(int)


    rows = df_proc.iloc[time_idx]

    grid_id = rows["grid_id"].values
    year = rows["year"].values
    month = rows["month"].values

    # === 7. 反标准化 ===
    gpp_mean = scaler.mean_[0]
    gpp_std = scaler.scale_[0]

    gpp_pred = preds_norm * gpp_std + gpp_mean
    gpp_true = y_true_norm * gpp_std + gpp_mean

    # === 8. 输出结果 ===
    out_df = pd.DataFrame({
        "grid_id": grid_id,
        "year": year,
        "month": month,
        "decoder_time_idx": time_idx,
        "gpp_true": gpp_true,
        "gpp_pred": gpp_pred,
    })
    # 按 grid_id / year / month 排序（仅影响显示顺序）
    out_df = out_df.sort_values(
        by=["grid_id", "year", "month"]
    ).reset_index(drop=True)

    out_path = os.path.join(
        project_root,
        "results",
        "predictions",
        "factual_predictions_full.csv",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"[INFO] Saved → {out_path}")
    print(f"[INFO] Rows: {len(out_df)}")


if __name__ == "__main__":
    main()
