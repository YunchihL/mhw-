"""
run_all_evaluations.py
=====================

统一的模型评估运行入口（A1–A4）

本脚本用于调度并运行四个相互独立的模型评估步骤：
A1–A4 用于从不同角度验证 TFT 模型的可靠性及其用于反事实模拟的合理性。

------------------------------------------------------------
评估步骤说明
------------------------------------------------------------

A1 (-a1): 整体预测性能验收（factual）
    - 计算 R² / MAE / RMSE
    - 判断本次模型训练是否“可接受”
    - 不涉及反事实、不做可视化

A2 (-a2): 时间结构一致性检验
    - 按低 / 中 / 高纬度带，各随机选取 1 个 grid（固定随机种子）
    - 展示连续 24 个月（月度，两年）的
      GPP_true vs GPP_pred_factual 时间序列
    - 用于直观判断模型是否学习到季节变化与年际节律

A3 (-a3): 惰性检验（Inertness check）
    - 比较 non-MHW 与 MHW 月份的 |ΔGPP%| 分布
    - 验证模型在“无事件”时期不会产生虚假响应
    - 是反事实模拟可信度的关键检验之一

A4 (-a4): 剂量–反应检验（Dose–response check）
    - 仅在 MHW 月份内分析
    - 直接使用 data.csv 中的强度密度（intensity_density）作为剂量指标
    - 检验 ΔGPP% 是否随剂量单调变化
    - 提供分箱统计与 Spearman 相关系数

------------------------------------------------------------
使用方式
------------------------------------------------------------

【默认行为】
不指定任何 -aX 参数时，默认依次运行 A1–A4 全部步骤：

    python -m code.evaluate.run_all_evaluations \
        --pred-csv results/predictions/factual_rolling_predictions.csv \
        --data-csv data/data.csv \
        --out-dir results

【指定运行某些步骤】
可以通过 -a1 / -a2 / -a3 / -a4 任意组合指定要运行的步骤：

    只运行 A1：
        -a1

    运行 A2 和 A4：
        -a2 -a4

示例：
    python -m code.evaluate.run_all_evaluations \
        --pred-csv results/predictions/factual_rolling_predictions.csv \
        --data-csv data/data.csv \
        --out-dir results \
        -a2 -a4

------------------------------------------------------------
参数说明
------------------------------------------------------------

--pred-csv
    A2–A4 共用的滚动事实预测 CSV 路径

--data-csv
    原始 data.csv（A3 / A4 需要 isMHW 与 intensity_density）

--out-dir
    A3 / A4 的输出根目录（自动创建 figures/ 与 tables/）
    A2 会输出到 out-dir/figures

--ckpt
    A1 的模型检查点（可选，默认自动选择 checkpoints/ 最新 ckpt）

--a1-interpret
    A1 追加特征重要性输出

--a2-seed / --a2-window
    A2 的随机种子与时间窗长度

--a3-seed / --a3-eps / --a3-clip-pct
    A3 的 bootstrap 种子与数值稳定参数

--a4-frac
    A4 LOWESS 平滑参数

------------------------------------------------------------
设计原则
------------------------------------------------------------

- A1–A4 各自为独立、可单独运行的脚本
- 本文件仅负责“调度”，不包含具体分析逻辑
- 所有结果均可通过 python -m 方式复现
- 适用于模型调试、结果筛选及论文复现

"""
