# ============================================================
# debug_time_mapping.py
#
# Sanity check for time mapping:
#   (grid_id, decoder_time_idx)  ->  (year, month)
# ============================================================

import os
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


def main():
    # --------------------------------------------------
    # 1. load data (same as training / rolling)
    # --------------------------------------------------
    config = get_config()
    df_raw = get_raw_data(config)
    df_proc, _ = preprocess(df_raw, config)

    df_proc = (
        df_proc
        .sort_values(["grid_id", "year", "month"])
        .reset_index(drop=True)
    )
    df_proc["time_idx"] = df_proc.groupby("grid_id").cumcount()

    print("\n=== df_proc check ===")
    print(df_proc[["grid_id", "year", "month", "time_idx"]].head(10))

    # --------------------------------------------------
    # 2. build rolling dataset
    # --------------------------------------------------
    training, _ = create_datasets(df_proc, config)

    rolling_ds = TimeSeriesDataSet.from_dataset(
        training,
        df_proc,
        stop_randomization=True,
    )

    rolling_dl = rolling_ds.to_dataloader(
        train=False,
        batch_size=128,
        num_workers=0,
    )

    # --------------------------------------------------
    # 3. load model (no need to be perfect)
    # --------------------------------------------------
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    ckpt_dir = os.path.join(project_root, "checkpoints")
    ckpt = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")])[-1]

    model = TemporalFusionTransformer.load_from_checkpoint(
        os.path.join(ckpt_dir, ckpt),
        strict=False,
    )
    model.eval()

    # --------------------------------------------------
    # 4. one batch predict (only for x)
    # --------------------------------------------------
    out = model.predict(
        rolling_dl,
        return_x=True,
        mode="prediction",
    )

    x = out[1]

    # --------------------------------------------------
    # 5. extract keys
    # --------------------------------------------------
    grid_id_encoded = x["groups"][:, 0].detach().cpu().numpy()
    decoder_time_idx = x["decoder_time_idx"][:, 0].detach().cpu().numpy()

    grid_id_encoder = rolling_ds.categorical_encoders["grid_id"]
    grid_id = grid_id_encoder.inverse_transform(grid_id_encoded).astype(str)

    print("\n=== extracted from x ===")
    print("grid_id sample:", grid_id[:10])
    print("decoder_time_idx sample:", decoder_time_idx[:10])

    # --------------------------------------------------
    # 6. mapping via df_proc
    # --------------------------------------------------
    lookup = (
        df_proc[["grid_id", "time_idx", "year", "month"]]
        .drop_duplicates()
        .set_index(["grid_id", "time_idx"])
    )

    idx = pd.MultiIndex.from_arrays(
        [grid_id, decoder_time_idx],
        names=["grid_id", "time_idx"]
    )

    # ---- sanity checks ----
    print("\n=== mapping sanity check ===")

    missing = ~idx.isin(lookup.index)
    print(f"Missing mappings: {missing.sum()} / {len(idx)}")

    if missing.any():
        print("Example missing keys:")
        print(idx[missing][:10])
        raise RuntimeError("❌ time mapping FAILED")

    year = lookup.loc[idx, "year"].values
    month = lookup.loc[idx, "month"].values

    print("\nMapped year/month sample:")
    for i in range(10):
        print(f"{grid_id[i]}  time_idx={decoder_time_idx[i]}  ->  {year[i]}-{month[i]:02d}")

    print("\n✅ TIME MAPPING SANITY CHECK PASSED")


if __name__ == "__main__":
    main()
