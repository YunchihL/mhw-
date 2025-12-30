📘 README（当前阶段版本 · 精简且准确）
🌱 红树林 GPP 预测 — Temporal Fusion Transformer (TFT)

本项目用于训练和验证 Temporal Fusion Transformer (TFT) 模型，目标是利用多源气候与海洋热浪（MHW）指标预测红树林的 月尺度总初级生产力 GPP（gC/month）。

当前版本已经实现：

数据预处理（标准化、特征工程、时间编码等）

TFT 模型训练（PyTorch Forecasting + Lightning 2.x）

模型预测（验证集）

标准化与反标准化指标评估（MAE / RMSE / R² / MAPE）

模型可视化（scatter plot）

一套清晰的项目结构

未来会进一步扩展为：

反事实模拟（无 MHW 情况）

CMIP6 未来预测

蓝碳损失评估

期刊可复现性打包

1. 项目结构（当前状态）
project/
│
├── config/
│   └── config.yaml         # 配置文件（模型参数 / 数据路径）
│
├── data/
│   └── data.csv            # 最终训练数据
│
├── code/
│   ├── train/              # 训练模块
│   │   └── train_tft.py
│   │
│   ├── evaluate/           # 评估模块
│   │   ├── factual_predict.py
│   │   ├── factual_rolling_predict.py
│   │   ├── a1_factual_metrics.py
│   │   ├── a2_latband_timeseries.py
│   │   ├── a3_inertness_check.py
│   │   ├── a4_dose_response_check.py
│   │   └── run_all_evaluations.py
│   │
│   ├── utils/              # 工具函数
│   │   └── data_utils.py
│
├── checkpoints/            # 模型权重（ckpt）
├── results/                # 预测 / 指标 / 图表输出
├── logs/                   # TensorBoard 日志
├── lightning_logs/         # Lightning 默认日志目录
│
└── requirements.txt        # 依赖文件

⚙️ 2. 环境安装
conda create -n tft python=3.10
conda activate tft
pip install -r requirements.txt


环境中包含：

PyTorch 2.3.1 + CUDA 12.1

Lightning 2.2.1

PyTorch Forecasting 1.5.0（与 Lightning 2.x 兼容）

numpy / pandas / sklearn / matplotlib

🚀 3. 数据说明（当前阶段）

数据文件 data/data.csv 已预处理好：

包含 CMIP6 可直接获取或可计算 的气候变量

包含海洋热浪相关能量指标

包含 NDVI，但目前未在训练中使用

预处理阶段自动生成：

month_sin / month_cos

标准化后的 gpp

time_idx

🏋️ 4. 模型训练

运行：

python -m code.train.train_tft --config config/config.yaml


输出：

code/best_tft-epoch=XX-val_loss=XXXX.ckpt

TensorBoard 日志：logs/tft/

查看训练曲线：

tensorboard --logdir logs/tft

🔍 5. 模型预测（验证集）
python -m code.evaluate.factual_predict \
    --ckpt checkpoints/tft-epoch=14-val_loss=0.1356.ckpt

滚动一步预测：
python -m code.evaluate.factual_rolling_predict \
    --ckpt checkpoints/tft-epoch=14-val_loss=0.1356.ckpt \
    --batch-size 256

📊 6. A1–A4 评估（可信度检查）
运行单个脚本：
python -m code.evaluate.a1_factual_metrics --ckpt checkpoints/tft-epoch=14-val_loss=0.1356.ckpt --interpret
python -m code.evaluate.a2_latband_timeseries --csv results/predictions/factual_rolling_predictions.csv --out-dir results/figures
python -m code.evaluate.a3_inertness_check --pred results/predictions/factual_rolling_predictions.csv --data data/data.csv --out-dir results
python -m code.evaluate.a4_dose_response_check --pred-csv results/predictions/factual_rolling_predictions.csv --data-csv data/data.csv --out-dir results

一键运行（可用 -a1/-a2/-a3/-a4 指定步骤）：
python -m code.evaluate.run_all_evaluations \
    --pred-csv results/predictions/factual_rolling_predictions.csv \
    --data-csv data/data.csv \
    --out-dir results

🧪 7. Dataloader 调试
python -m code.utils.test_batch_format


用于检查 validation batch 的格式。

📌 8. 注意事项（当前阶段）

本版本尚未包含 反事实模拟 或 未来 CMIP6 预测，后续将添加 simulate_counterfactual.py。

NDVI 已从训练特征中移除，因为 CMIP6 无未来 NDVI。

所有超参数应在 config.yaml 修改，而不是在代码中修改。
