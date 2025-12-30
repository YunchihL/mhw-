# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a scientific research project for **mangrove GPP (Gross Primary Productivity) prediction** using **Temporal Fusion Transformer (TFT)** models. The project focuses on climate science applications, specifically predicting mangrove carbon uptake using climate and marine heatwave (MHW) data.

## Architecture

### Core Framework
- **PyTorch Lightning** for training orchestration
- **PyTorch Forecasting** for TFT implementation
- **TensorBoard** for visualization
- Centralized configuration in `config/config.yaml`

### Directory Structure
- `code/train/` - Training module (`train_tft.py`)
- `code/evaluate/` - Comprehensive evaluation framework (A1-A4 checks)
- `code/utils/` - Shared utilities (`data_utils.py`, `test_batch_format.py`)
- `config/` - Centralized configuration (`config.yaml`)
- `data/` - Input datasets (`data.csv`, `data_counterfactual.csv`)
- `checkpoints/` - Model weights and scaler
- `results/` - Organized outputs (predictions, metrics, figures, tables, plots)
- `logs/` - TensorBoard logs
- `lightning_logs/` - PyTorch Lightning logs (77+ versions)

### Model Architecture
- Temporal Fusion Transformer (TFT) for multi-horizon time series forecasting
- Encoder: 12 months historical context
- Decoder: 3 months future prediction
- Features: Climate variables + MHW indicators + spatial metadata

### Evaluation Framework (A1-A4)
- **A1**: Factual prediction performance + feature importance
- **A2**: Latitude-band time series analysis
- **A3**: Inertness check (model stability)
- **A4**: Dose-response relationship validation
- Supports both factual and counterfactual scenarios

## Common Development Commands

### Environment Setup
```bash
conda create -n tft python=3.10
conda activate tft
pip install -r requirements.txt  # CUDA-enabled PyTorch stack
```

### Training
```bash
# Full training
python -m code.train.train_tft --config config/config.yaml

# Smoke test (1 epoch)
python test_train_tft.py

# View training curves
tensorboard --logdir logs/tft
```

### Prediction
```bash
# Factual prediction
python -m code.evaluate.factual_predict --ckpt checkpoints/tft-epoch=14-val_loss=0.1356.ckpt

# Rolling one-step-ahead prediction
python -m code.evaluate.factual_rolling_predict --ckpt checkpoints/tft-epoch=14-val_loss=0.1356.ckpt --batch-size 256

# Counterfactual prediction
python -m code.evaluate.counterfactual_rolling_predict --cf-data data/data_counterfactual.csv --ckpt checkpoints/tft-epoch=14-val_loss=0.1356.ckpt
```

### Evaluation (A1-A4)
```bash
# Individual checks
python -m code.evaluate.a1_factual_metrics --ckpt checkpoints/tft-epoch=14-val_loss=0.1356.ckpt --interpret
python -m code.evaluate.a2_latband_timeseries --csv results/predictions/factual_rolling_predictions.csv --out-dir results/figures
python -m code.evaluate.a3_inertness_check --pred results/predictions/factual_rolling_predictions.csv --data data/data.csv --out-dir results
python -m code.evaluate.a4_dose_response_check --pred-csv results/predictions/factual_rolling_predictions.csv --data-csv data/data.csv --out-dir results

# Run all evaluations
python -m code.evaluate.run_all_evaluations --pred-csv results/predictions/factual_rolling_predictions.csv --data-csv data/data.csv --out-dir results

# Run selected checks
python -m code.evaluate.run_all_evaluations --input results/predictions/counterfactual_rolling_predictions.csv --out-dir results/figures -a1 -a3
```

### Data Preparation
```bash
# Create counterfactual data
python -m code.evaluate.build_counterfactual_data --data data/data.csv --unreal-sst data/grid_monthly_unreal_sst_final.csv --out data/data_counterfactual.csv
```

### Testing & Debugging
```bash
# Dataloader format validation
python -m code.utils.test_batch_format
```

## Configuration

### Key Configuration (`config/config.yaml`)
- Model: `encoder_length=12`, `prediction_length=3`, `hidden_size=64`
- Training: `batch_size=32`, `max_epochs=30`, `learning_rate=0.001`
- Data paths and optimization settings

**Important**: All hyperparameters should be modified in `config/config.yaml`, not in code.

## Data Pipeline

### Input Data
- `data/data.csv` - Primary dataset with climate variables and MHW indicators
- `data/data_counterfactual.csv` - Counterfactual dataset (no MHW)

### Preprocessing (`data_utils.py`)
- Standardized preprocessing pipeline
- Month encoding (sin/cos transformations)
- Target normalization (gpp_total)
- Grid-based time series organization

## Output Organization

### Checkpoints
- `checkpoints/` - Model weights (`.ckpt`) and scaler (`.pkl`)
- Example: `checkpoints/tft-epoch=14-val_loss=0.1356.ckpt`

### Results Structure
- `results/predictions/` - Prediction CSVs
- `results/metrics/` - Evaluation metrics (JSON, TXT)
- `results/figures/` - Generated plots
- `results/tables/` - Result tables
- `results/plots/` - Additional visualizations

### Logs
- `logs/tft/` - TensorBoard logs for visualization
- `lightning_logs/` - PyTorch Lightning training logs (versioned)

## Development Guidelines

### Code Style
- Python only with 4-space indentation
- `snake_case` for functions/variables, `UPPER_SNAKE_CASE` for constants
- Entry points runnable as modules (`python -m code.<package>.<script>`)
- No enforced formatter/linter; maintain consistency with existing style

### Testing
- Lightweight scripts instead of test framework
- `test_train_tft.py` - 1-epoch smoke test of training pipeline
- `code/utils/test_batch_format.py` - Dataloader batch structure validation

### Reproducibility
- Centralize hyperparameters in `config/config.yaml`
- Keep paths relative to repo root
- Outputs organized under `results/` and `logs/`

## Notes for Future Development

1. **Counterfactual Simulation**: Future work includes `simulate_counterfactual.py` for MHW-free scenarios
2. **CMIP6 Future Prediction**: Planned extension for climate model projections
3. **Blue Carbon Loss Assessment**: Future analysis module
4. **Journal Reproducibility**: Structure designed for publication-ready code

The project is research-grade with clear separation of concerns, configuration-driven training, and comprehensive evaluation framework.