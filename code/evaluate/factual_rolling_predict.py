# ============================================================
# factual_rolling_predict.py
#
# 全时间 rolling one-step-ahead factual prediction
# 输出：每个 (grid_id, year, month) 的
#       gpp_true / gpp_pred
#
# 用于：
#   - A1：整体预测指标
#   - A2：时间结构一致性（24 个月）
#
# ✅ 完全遵循 PyTorch-Forecasting 官方推荐方式
# ============================================================

import os
import argparse
import numpy as np
import pandas as pd
import torch

from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet

from code.train.train_tft import (
    get_config,
    get_raw_data,
    preprocess,
    create_datasets,
)


# ------------------------------------------------------------
# 工具：致命错误（不再猜）
# ------------------------------------------------------------
def FATAL(msg):
    raise RuntimeError(f"\n[FATAL] {msg}\n")


# ------------------------------------------------------------
# ckpt 路径解析
# ------------------------------------------------------------
def resolve_ckpt_path(ckpt_arg: str | None):
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    ckpt_dir = os.path.join(project_root, "checkpoints")

    if ckpt_arg is None:
        ckpts = sorted(
            [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")],
            reverse=True,
        )
        if not ckpts:
            FATAL("checkpoints/ 中没有 ckpt")
        print(f"[INFO] 使用最新 ckpt: {ckpts[0]}")
        return os.path.join(ckpt_dir, ckpts[0])

    if os.path.exists(ckpt_arg):
        return ckpt_arg

    candidate = os.path.join(ckpt_dir, ckpt_arg)
    if os.path.exists(candidate):
        return candidate

    FATAL(f"找不到 ckpt: {ckpt_arg}")


# ------------------------------------------------------------
# month 从 sin / cos 反推（稳妥）
# ------------------------------------------------------------
def month_from_sin_cos(month_sin, month_cos):
    angle = np.arctan2(month_sin, month_cos)
    month = (np.round(angle / (2 * np.pi) * 12).astype(int) % 12) + 1
    return month


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument(
        "--out",
        type=str,
        default="results/predictions/factual_rolling_predictions.csv",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    # --------------------------------------------------
    # 1. checkpoint
    # --------------------------------------------------
    ckpt_path = resolve_ckpt_path(args.ckpt)
    print(f"[INFO] Using checkpoint: {ckpt_path}")

    # --------------------------------------------------
    # 2. data & preprocess
    # --------------------------------------------------
    config = get_config()
    df_raw = get_raw_data(config)
    # df_raw = pd.read_csv("data/data_counterfactual.csv")

    df_proc, scaler = preprocess(df_raw, config)

    # ⚠️ 与训练阶段完全一致的 time_idx
    df_proc = (
        df_proc
        .sort_values(["grid_id", "year", "month"])
        .reset_index(drop=True)
    )
    df_proc["time_idx"] = df_proc.groupby("grid_id").cumcount()

    # grid_id 强制 int（后面 merge 用）
    df_proc["grid_id"] = df_proc["grid_id"].astype(str)

    # --------------------------------------------------
    # 3. datasets
    # --------------------------------------------------
    training, _ = create_datasets(df_proc, config)

    # ⭐ 官方推荐：from_dataset + stop_randomization
    rolling_ds = TimeSeriesDataSet.from_dataset(
        training,
        df_proc,
        stop_randomization=True,
    )

    rolling_dl = rolling_ds.to_dataloader(
        train=False,
        batch_size=args.batch_size,
        num_workers=0,
    )

    # --------------------------------------------------
    # 4. model
    # --------------------------------------------------
    model = TemporalFusionTransformer.load_from_checkpoint(
        ckpt_path, strict=False
    )
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # --------------------------------------------------
    # 5. predict（官方方式）
    # --------------------------------------------------
    out = model.predict(
        rolling_dl,
        return_x=True,
        mode="prediction",
    )

    # PyTorch-Forecasting 的 Prediction 对象
    preds = out[0]
    x = out[1]

    preds = preds.detach().cpu().numpy()

    # preds 可能是 [N, pred_len] 或 [N, pred_len, 1]
    if preds.ndim == 3:
        gpp_pred_norm = preds[:, 0, 0]
    elif preds.ndim == 2:
        gpp_pred_norm = preds[:, 0]
    else:
        FATAL(f"preds 维度异常: {preds.shape}")

    # decoder 真实值（第 1 步）
    gpp_true_norm = (
        x["decoder_target"][:, 0]
        .detach()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------
    # 6. meta 信息（完全来自 x）
    # --------------------------------------------------
    grid_id_encoded = (
        x["groups"][:, 0]
        .detach()
        .cpu()
        .numpy()
    )

    # categorical encoder（官方方式）
    grid_id_encoder = rolling_ds.categorical_encoders["grid_id"]

    # encoded -> 原始 grid_id
    grid_id = grid_id_encoder.inverse_transform(grid_id_encoded)
    grid_id = grid_id.astype(str)



        # --------------------------------------------------
    # ✅ year/month：用 decoder_time_idx 回查 df_proc（最稳）
    # --------------------------------------------------
    decoder_time_idx = (
        x["decoder_time_idx"][:, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(int)
    )

    time_lookup = (
        df_proc[["grid_id", "time_idx", "year", "month"]]
        .drop_duplicates()
        .set_index(["grid_id", "time_idx"])
        .sort_index()
    )

    key = pd.MultiIndex.from_arrays(
        [grid_id, decoder_time_idx],
        names=["grid_id", "time_idx"],
    )

    # 防御性检查：必须 100% 能映射
    missing = ~key.isin(time_lookup.index)
    if missing.any():
        bad = key[missing][:10]
        FATAL(
            "year/month 映射失败：存在 (grid_id, decoder_time_idx) 不在 df_proc 中。\n"
            f"示例: {list(bad)}\n"
            "请确认：df_proc 的 time_idx 构建方式与训练阶段一致，且 grid_id dtype 一致。"
        )

    year = time_lookup.loc[key, "year"].to_numpy().astype(int)
    month = time_lookup.loc[key, "month"].to_numpy().astype(int)


    # --------------------------------------------------
    # 7. lat_c：直接从 df_proc 映射（官方也这么做）
    # --------------------------------------------------
    lat_lookup = (
        df_proc[["grid_id", "lat_c"]]
        .drop_duplicates()
        .set_index("grid_id")["lat_c"]
    )

    # 防御性检查
    missing = np.setdiff1d(grid_id, lat_lookup.index.values)
    if len(missing) > 0:
        FATAL(f"有 grid_id 找不到 lat_c，例如: {missing[:10]}")

    lat_c = lat_lookup.loc[grid_id].values

    # --------------------------------------------------
    # 8. inverse scaling
    # --------------------------------------------------
    gpp_mean = scaler.mean_[0]
    gpp_std = scaler.scale_[0]

    gpp_true = gpp_true_norm * gpp_std + gpp_mean
    gpp_pred = gpp_pred_norm * gpp_std + gpp_mean

    # --------------------------------------------------
    # 9. output
    # --------------------------------------------------
    out_df = pd.DataFrame({
        "grid_id": grid_id.astype(int),
        "year": year.astype(int),
        "month": month.astype(int),
        "lat_c": lat_c,
        "gpp_true": gpp_true,
        "gpp_pred": gpp_pred,
    })

    # 排序 + 去重（rolling 可能多次命中同一月）
    out_df = (
        out_df
        .sort_values(["grid_id", "year", "month"])
        .drop_duplicates(
            subset=["grid_id", "year", "month"],
            keep="first"
        )
        .reset_index(drop=True)
    )

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(project_root, out_path)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"[INFO] Saved → {out_path}")
    print(f"[INFO] Rows: {len(out_df)}")


if __name__ == "__main__":
    main()
