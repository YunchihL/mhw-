# ============================================================
# A1: Factual Prediction Performance + Feature Importance
# ============================================================

import argparse
import os

# Force CPU in environments where CUDA is present but unusable.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PL_DISABLE_FABRIC_CUDA", "1")
import numpy as np
import pandas as pd

import torch
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE

from code.train.train_tft import (
    get_config,
    get_raw_data,
    preprocess,
    create_datasets,
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

    config = get_config()
    df_raw = get_raw_data(config)
    df_proc, scaler = preprocess(df_raw, config)
    training, validation = create_datasets(df_proc, config)

    model_cfg = config["model"]
    loss = MAE() if model_cfg.get("loss", "MAE") == "MAE" else None
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=model_cfg["learning_rate"],
        hidden_size=model_cfg["hidden_size"],
        attention_head_size=model_cfg["attention_head_size"],
        dropout=model_cfg["dropout"],
        hidden_continuous_size=model_cfg["hidden_continuous_size"],
        lstm_layers=model_cfg["lstm_layers"],
        loss=loss,
        log_interval=10,
        log_val_interval=1,
        reduce_on_plateau_patience=3,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(
        ckpt["state_dict"], strict=False
    )
    if missing:
        print(f"[WARN] Missing keys in state_dict: {missing[:5]}")
    if unexpected:
        print(f"[WARN] Unexpected keys in state_dict: {unexpected[:5]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
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
