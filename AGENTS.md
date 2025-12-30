# Repository Guidelines

## Project Structure & Module Organization
- `code/train/`: training entry points (e.g., `train_tft.py`) and model setup.
- `code/evaluate/`: evaluation and analysis scripts, including batch runs (`run_all_evaluations.py`).
- `code/utils/`: shared utilities (data loading, preprocessing, debug helpers).
- `config/config.yaml`: model/training configuration used by training scripts.
- `data/`: input dataset(s), primarily `data.csv`.
- `results/`, `logs/`, `lightning_logs/`, `checkpoints/`: outputs, TensorBoard logs, and saved weights.

## Build, Test, and Development Commands
- `pip install -r requirements.txt`: install Python dependencies (CUDA-specific PyTorch wheels included).
- `python -m code.train.train_tft --config config/config.yaml`: train TFT model (defaults to `config/config.yaml`).
- `tensorboard --logdir logs/tft`: inspect training curves.
- `python -m code.evaluate.factual_predict --ckpt checkpoints/tft-epoch=14-val_loss=0.1356.ckpt`: full factual prediction output.
- `python -m code.evaluate.factual_rolling_predict --ckpt checkpoints/tft-epoch=14-val_loss=0.1356.ckpt --batch-size 256`: rolling one-step-ahead factual predictions.
- `python -m code.evaluate.a1_factual_metrics --ckpt checkpoints/tft-epoch=14-val_loss=0.1356.ckpt --interpret`: metrics plus feature importance plots/tables.
- `python -m code.evaluate.build_counterfactual_data --data data/data.csv --unreal-sst data/grid_monthly_unreal_sst_final.csv --out data/data_counterfactual.csv`: create counterfactual inputs.
- `python -m code.evaluate.counterfactual_rolling_predict --cf-data data/data_counterfactual.csv --ckpt checkpoints/tft-epoch=14-val_loss=0.1356.ckpt`: rolling counterfactual predictions.
- `python -m code.evaluate.run_all_evaluations --input results/predictions/counterfactual_rolling_predictions.csv --out-dir results/figures -a1 -a3`: run selected A1–A4 checks.

## Common Paths
- Checkpoints: `checkpoints/` (e.g., `checkpoints/tft-epoch=14-val_loss=0.1356.ckpt`).
- Predictions: `results/predictions/` (e.g., `factual_rolling_predictions.csv`).
- Metrics and tables: `results/metrics/`, `results/tables/`.
- Figures: `results/figures/`, `results/plots/`.
- Logs: `logs/tft/` and `lightning_logs/`.

## Coding Style & Naming Conventions
- Python only; use 4-space indentation and PEP 8 style.
- Prefer `snake_case` for functions/variables, `UPPER_SNAKE_CASE` for constants.
- Keep entry points runnable as modules (`python -m code.<package>.<script>`).
- No formatter/linter enforced; keep changes minimal and consistent with surrounding style.

## Testing Guidelines
- Lightweight scripts are used instead of a test framework.
- `test_train_tft.py` performs a 1-epoch smoke test of the training pipeline.
- `code/utils/test_batch_format.py` validates dataloader batch structure.
- Naming pattern: `test_*.py`; run via `python test_train_tft.py` or `python -m code.utils.test_batch_format`.

## Commit & Pull Request Guidelines
- This repository does not include Git history; no formal commit conventions are defined.
- If using Git, prefer concise, imperative messages with scope tags (e.g., `train: adjust encoder length`).
- PRs should include: purpose, key commands run, and any new artifacts/figures produced.

## Configuration & Reproducibility Notes
- Centralize hyperparameters and paths in `config/config.yaml`; avoid hardcoding.
- Outputs are written under `results/` and `logs/`; keep paths relative to repo root.
