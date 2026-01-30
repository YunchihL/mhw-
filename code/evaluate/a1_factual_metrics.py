# ============================================================
# A1: Factual Prediction Performance + Feature Importance
# ============================================================

import argparse
import os
import sys

# 解决与标准库code模块的命名冲突
# 移除当前目录和项目根目录，避免干扰标准库导入
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

# 临时从sys.path中移除可能引起冲突的路径
paths_to_remove = []
for path in sys.path:
    if path == '' or path == '.' or path == project_root:
        paths_to_remove.append(path)
for path in paths_to_remove:
    sys.path.remove(path)

# 添加项目根目录的父目录，这样仍然可以导入code模块
parent_of_root = os.path.dirname(project_root)
if parent_of_root not in sys.path:
    sys.path.insert(0, parent_of_root)

# Priority: use CUDA if available, fallback to CPU
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
# os.environ.setdefault("PL_DISABLE_FABRIC_CUDA", "1")
import numpy as np
import pandas as pd

import torch
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

from pytorch_forecasting.models import TemporalFusionTransformer
from code.train.train_tft import (
    get_config,
    get_raw_data,
    preprocess,
)


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
            raise FileNotFoundError("❌ checkpoints/ 中没有 ckpt 文件")
        print(f"[INFO] 使用最新 ckpt: {ckpts[0]}")
        return os.path.join(ckpt_dir, ckpts[0])

    if os.path.exists(ckpt_arg):
        return ckpt_arg

    candidate = os.path.join(ckpt_dir, ckpt_arg)
    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError(f"❌ 找不到 ckpt: {ckpt_arg}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--interpret", action="store_true")
    args = parser.parse_args()

    ckpt_path = resolve_ckpt_path(args.ckpt)
    print(f"[INFO] Using checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    dataset_params = ckpt.get("dataset_parameters")

    config = get_config()
    df_raw = get_raw_data(config)
    df_proc, scaler = preprocess(df_raw, config)

    df_proc = df_proc.sort_values(
        ["grid_id", "year", "month"]
    ).reset_index(drop=True)
    df_proc["time_idx"] = df_proc.groupby("grid_id").cumcount()

    if dataset_params:
        max_pred_len = dataset_params.get("max_prediction_length")
        if max_pred_len is None:
            raise RuntimeError("dataset_parameters missing max_prediction_length")

        max_idx = df_proc["time_idx"].max()
        validation_start = max_idx - max_pred_len + 1
        train_df = df_proc[df_proc["time_idx"] < validation_start]

        training = TimeSeriesDataSet.from_parameters(
            dataset_params,
            train_df,
            stop_randomization=True,
        )
        validation = TimeSeriesDataSet.from_dataset(
            training,
            df_proc,
            predict=True,
            stop_randomization=True,
        )
    else:
        from code.train.train_tft import create_datasets

        training, validation = create_datasets(df_proc, config)

    hparams = ckpt.get("hyper_parameters", {})
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=hparams.get("learning_rate", 0.001),
        hidden_size=hparams.get("hidden_size", 64),
        attention_head_size=hparams.get("attention_head_size", 4),
        dropout=hparams.get("dropout", 0.1),
        hidden_continuous_size=hparams.get("hidden_continuous_size", 16),
        lstm_layers=hparams.get("lstm_layers", 2),
        loss=hparams.get("loss", MAE()),
        log_interval=hparams.get("log_interval", 10),
        log_val_interval=hparams.get("log_val_interval", 1),
        reduce_on_plateau_patience=hparams.get("reduce_on_plateau_patience", 3),
    )

    missing, unexpected = model.load_state_dict(
        ckpt["state_dict"], strict=False
    )
    if missing:
        print(f"[WARN] Missing keys in state_dict: {missing[:5]}")
    if unexpected:
        print(f"[WARN] Unexpected keys in state_dict: {unexpected[:5]}")

    model.eval()

    # Avoid multiprocessing in restricted environments.
    num_workers = 0
    val_dl = validation.to_dataloader(
        train=False, batch_size=64, num_workers=num_workers
    )

    out = model.predict(
        val_dl,
        return_x=True,
        mode="prediction",
    )

    preds = out[0].detach().cpu().numpy()
    x = out[1]

    y_pred = preds[:, 0] if preds.ndim == 2 else preds[:, 0, 0]
    y_true = x["decoder_target"][:, 0].detach().cpu().numpy()

    gpp_mean = scaler.mean_[0]
    gpp_std = scaler.scale_[0]
    y_true = y_true * gpp_std + gpp_mean
    y_pred = y_pred * gpp_std + gpp_mean

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print("\n========== A1: Factual Prediction Performance ==========")
    print(f"N samples : {len(y_true):,}")
    print(f"R²        : {r2:.4f}")
    print(f"MAE       : {mae:.4e}")
    print(f"RMSE      : {rmse:.4e}")
    print("========================================================\n")

    if not args.interpret:
        return

    raw_out = model.predict(val_dl, mode="raw", return_x=True)
    raw_predictions = raw_out[0]

    interpretation = model.interpret_output(
        raw_predictions, reduction="mean"
    )

    print("[DEBUG] interpretation keys:", interpretation.keys())

    rows = []

    def append_block(weights, names, group):
        if weights is None:
            return
        weights = weights.detach().cpu().numpy()
        names = list(names)

        if len(weights) != len(names):
            print(
                f"[WARN] Length mismatch in {group}: "
                f"{len(weights)} weights vs {len(names)} names. "
                f"Trimming names to match weights."
            )
            names = names[-len(weights):]

        for n, w in zip(names, weights):
            rows.append({
                "variable": n,
                "group": group,
                "importance": float(w),
            })

    append_block(
        interpretation.get("encoder_variables"),
        model.hparams.time_varying_reals_encoder,
        "encoder",
    )

    append_block(
        interpretation.get("decoder_variables"),
        model.hparams.time_varying_reals_decoder,
        "decoder",
    )

    append_block(
        interpretation.get("static_variables"),
        model.hparams.static_reals,
        "static",
    )

    imp_df = (
        pd.DataFrame(rows)
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    table_dir = os.path.join(project_root, "results", "tables")
    fig_dir = os.path.join(project_root, "results", "figures")
    os.makedirs(table_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    table_path = os.path.join(
        table_dir, "A1_feature_importance.csv"
    )
    imp_df.to_csv(table_path, index=False)
    print(f"[INFO] Saved feature importance table → {table_path}")

    fig_dict = model.plot_interpretation(interpretation)
    for name, fig in fig_dict.items():
        fig_path = os.path.join(
            fig_dir, f"A1_feature_importance_{name}.png"
        )
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved feature importance figure → {fig_path}")


if __name__ == "__main__":
    main()
