命令大全（项目内可直接复制使用）

1) 环境安装
```bash
pip install -r requirements.txt
```

2) 训练
```bash
python -m code.train.train_tft --config config/config.yaml
```

3) 事实预测（全量）
```bash
python -m code.evaluate.factual_predict --ckpt checkpoints/tft-epoch=14-val_loss=0.1356.ckpt
```

4) 事实滚动一步预测（A1/A2/A3/A4 使用）
```bash
python -m code.evaluate.factual_rolling_predict \
  --ckpt checkpoints/tft-epoch=14-val_loss=0.1356.ckpt \
  --out results/predictions/factual_rolling_predictions.csv \
  --batch-size 256
```

5) A1-A4 单独运行
```bash
python -m code.evaluate.a1_factual_metrics --ckpt checkpoints/tft-epoch=14-val_loss=0.1356.ckpt --interpret
python -m code.evaluate.a2_latband_timeseries --csv results/predictions/factual_rolling_predictions.csv --out-dir results/figures --seed 1 --window 24
python -m code.evaluate.a3_inertness_check --pred results/predictions/factual_rolling_predictions.csv --data data/data.csv --out-dir results --eps 1e-6 --clip-pct 200 --seed 42
python -m code.evaluate.a4_dose_response_check --pred-csv results/predictions/factual_rolling_predictions.csv --data-csv data/data.csv --out-dir results --frac 0.3
```

6) A1-A4 一键运行（可用 -a1/-a2/-a3/-a4 指定步骤）
```bash
python -m code.evaluate.run_all_evaluations \
  --pred-csv results/predictions/factual_rolling_predictions.csv \
  --data-csv data/data.csv \
  --out-dir results \
  --ckpt checkpoints/tft-epoch=14-val_loss=0.1356.ckpt \
  --a1-interpret
```

7) 反事实数据构建
```bash
python -m code.evaluate.build_counterfactual_data \
  --data data/data.csv \
  --unreal-sst data/grid_monthly_unreal_sst_final.csv \
  --out data/data_counterfactual.csv
```

8) 反事实滚动一步预测
```bash
python -m code.evaluate.counterfactual_rolling_predict \
  --cf-data data/data_counterfactual.csv \
  --ckpt checkpoints/tft-epoch=14-val_loss=0.1356.ckpt \
  --out results/predictions/counterfactual_rolling_predictions.csv \
  --batch-size 256
```

9) A3 备用版（箱线图/分箱图）
```bash
python -m code.evaluate.a3_inertness_check_boxplot \
  --pred-csv results/predictions/factual_rolling_predictions.csv \
  --out-table results/tables/A3_inertness_boxplot_summary.csv \
  --out-fig results/figures/A3_inertness_abs_delta_pct_boxplot.png \
  --eps 1e-9
```

10) 调试与测试
```bash
python -m code.evaluate.debug_time_mapping
python -m code.utils.test_batch_format
python test_train_tft.py
```

11) TensorBoard
```bash
tensorboard --logdir logs/tft
```
