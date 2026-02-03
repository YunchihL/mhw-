# ============================================================
#  Temporal Fusion Transformer (TFT) Training Script - Experiment Version
#  Support multiple experiments with separate folders
# ============================================================

import os
import argparse
import yaml
import joblib
import shutil

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger

from pytorch_forecasting.data import TimeSeriesDataSet, GroupNormalizer
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE

# 💡 所有工具函数都从 code.utils 导入（专业结构）
from code.utils.data_utils import load_data, preprocess_data


# ------------------------------------------------------------
# 1. Read config.yaml with optional overrides
# ------------------------------------------------------------
def get_config(config_path: str | None = None, overrides: dict = None):
    if config_path is None:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(this_dir, "..", "..", "config", "config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Apply overrides if provided
    if overrides:
        for key_path, value in overrides.items():
            # Handle nested keys like "model.hidden_size"
            keys = key_path.split('.')
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value

    return config


# ------------------------------------------------------------
# 2. Load raw data
# ------------------------------------------------------------
def get_raw_data(config):
    return load_data(config)


# ------------------------------------------------------------
# 3. Preprocess + save scaler to experiment directory
# ------------------------------------------------------------
def preprocess(df, config, experiment_dir=None):
    df, scaler = preprocess_data(df, config)

    # ---- Patch: ensure categorical dtypes ----
    df["grid_id"] = df["grid_id"].astype(str)
    df["isMHW"]   = df["isMHW"].astype(str)

    # 保存 scaler 到实验目录或默认 checkpoints/
    if experiment_dir:
        ckpt_dir = os.path.join(experiment_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        scaler_path = os.path.join(ckpt_dir, "scaler.pkl")
    else:
        # 保存到项目根目录的 checkpoints/
        project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
        ckpt_dir = os.path.join(project_root, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        scaler_path = os.path.join(ckpt_dir, "scaler.pkl")

    joblib.dump(scaler, scaler_path)
    print(f"[INFO] Saved scaler → {scaler_path}")
    return df, scaler


# ------------------------------------------------------------
# 4. Build dataset for TFT
# ------------------------------------------------------------
def create_datasets(df, config):
    model_cfg = config["model"]
    encoder_length = model_cfg["encoder_length"]
    prediction_length = model_cfg["prediction_length"]

    df = df.sort_values(by=["grid_id", "year", "month"]).reset_index(drop=True)

    df["time_idx"] = df.groupby("grid_id").cumcount()
    max_idx = df["time_idx"].max()
    validation_start = max_idx - prediction_length + 1

    # ---------- Feature groups ----------
    static_categoricals = ["grid_id"]
    static_reals = ["lon_c", "lat_c", "mangrove_area"]

    time_varying_known_categoricals = ["isMHW"]

    time_varying_known_reals = [
        "year",
        "month_sin",
        "month_cos",
        "pr_mm",
        "tavg_celsius",
        "tmmx_celsius",
        "tmmn_celsius",
        "srad_mj_m2_day",
        "vpd_pa",
        "ws_mps",
        "mean_sst",
        "intensity_max_month",
        "duration_weighted_sum",
        "intensity_cumulative_weighted_sum",
        "intensity_density",
    ]

    time_varying_unknown_reals = ["gpp_total"]

    training = TimeSeriesDataSet(
        df[lambda x: x.time_idx < validation_start],
        time_idx="time_idx",
        target="gpp_total",
        group_ids=["grid_id"],

        min_encoder_length=encoder_length,
        max_encoder_length=encoder_length,
        min_prediction_length=prediction_length,
        max_prediction_length=prediction_length,

        static_categoricals=static_categoricals,
        static_reals=static_reals,
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,

        target_normalizer=GroupNormalizer(groups=["grid_id"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        df,
        predict=True,
        stop_randomization=True,
    )

    return training, validation


# ------------------------------------------------------------
# 5. Build TFT model
# ------------------------------------------------------------
def create_model(training, config):
    model_cfg = config["model"]

    # Loss
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

    print(model)
    return model


# ------------------------------------------------------------
# 6. Train with experiment directory support
# ------------------------------------------------------------
def run_training(model, training, validation, config, experiment_dir=None):
    train_cfg = config["training"]

    num_workers = int(train_cfg.get("num_workers", 0))
    train_loader = training.to_dataloader(
        train=True, batch_size=train_cfg["batch_size"], num_workers=num_workers
    )
    val_loader = validation.to_dataloader(
        train=False, batch_size=train_cfg["batch_size"], num_workers=num_workers
    )

    # Determine checkpoint directory
    if experiment_dir:
        ckpt_dir = os.path.join(experiment_dir, "checkpoints")
        log_dir = os.path.join(experiment_dir, "logs")
    else:
        # Use default directories
        project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
        ckpt_dir = os.path.join(project_root, "checkpoints")
        log_dir = "logs"  # TensorBoardLogger默认相对路径

    os.makedirs(ckpt_dir, exist_ok=True)
    if experiment_dir:
        os.makedirs(log_dir, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=train_cfg["early_stop_patience"]),
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="tft-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
    ]

    # Create logger - use experiment directory if provided
    if experiment_dir:
        logger = TensorBoardLogger(save_dir=experiment_dir, name="tft")
    else:
        logger = TensorBoardLogger(save_dir="logs", name="tft")

    trainer = pl.Trainer(
        max_epochs=train_cfg["max_epochs"],
        gradient_clip_val=train_cfg["gradient_clip_val"],
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader)
    return model, trainer


# ------------------------------------------------------------
# Main function for single experiment
# ------------------------------------------------------------
def train_experiment(config_path=None, experiment_dir=None, overrides=None):
    """
    Run a single training experiment with optional parameter overrides.

    Args:
        config_path: Path to config.yaml (default: project config)
        experiment_dir: Directory to save experiment results
        overrides: Dict of parameter overrides (e.g., {"model.hidden_size": 128})
    """
    # Create experiment directory if specified
    if experiment_dir:
        os.makedirs(experiment_dir, exist_ok=True)
        print(f"[INFO] Experiment directory: {experiment_dir}")

        # Save config with overrides
        config = get_config(config_path, overrides)
        config_save_path = os.path.join(experiment_dir, "config.yaml")
        with open(config_save_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"[INFO] Saved config → {config_save_path}")
    else:
        config = get_config(config_path, overrides)

    # Load and preprocess data
    df = get_raw_data(config)
    df, scaler = preprocess(df, config, experiment_dir)

    # Create datasets and model
    training, validation = create_datasets(df, config)
    model = create_model(training, config)

    # Train
    run_training(model, training, validation, config, experiment_dir)

    print(f"\n[INFO] Experiment finished. Results saved to: {experiment_dir or 'default locations'}\n")
    return model, scaler


# ------------------------------------------------------------
# Command line interface
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TFT Training Experiment")
    parser.add_argument("--config", type=str, help="Path to config.yaml")
    parser.add_argument("--experiment-dir", type=str, help="Directory to save experiment results")
    parser.add_argument("--overrides", type=str, nargs="+",
                       help="Parameter overrides in format key=value (e.g., model.hidden_size=128)")

    args = parser.parse_args()

    # Parse overrides if provided
    overrides = {}
    if args.overrides:
        for override in args.overrides:
            key, value = override.split("=", 1)
            # Try to convert value to appropriate type
            try:
                # Try int
                value = int(value)
            except ValueError:
                try:
                    # Try float
                    value = float(value)
                except ValueError:
                    # Keep as string
                    pass
            overrides[key] = value

    # Run experiment
    train_experiment(
        config_path=args.config,
        experiment_dir=args.experiment_dir,
        overrides=overrides
    )


if __name__ == "__main__":
    main()