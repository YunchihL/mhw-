# ============================================================
# test_train_tft.py
# 目的：快速测试 train_tft.py 是否能正常运行
# 不进行完整训练，只跑 1 epoch、batch_size=4
# ============================================================

import os
import yaml

# 导入 train_tft 的函数
from code.train.train_tft import (
    get_config,
    get_raw_data,
    preprocess,
    create_datasets,
    create_model,
    run_training,
)

# ------------------------------------------------------------
# 1. 创建一个临时 config（覆盖训练参数）
# ------------------------------------------------------------
def make_test_config():
    config = get_config()

    # 覆盖训练设置（快速跑）
    config["training"]["batch_size"] = 4
    config["training"]["max_epochs"] = 1
    config["training"]["early_stop_patience"] = 1
    config["training"]["gradient_clip_val"] = 0.1

    print("\n=== TEST CONFIG ===")
    print(yaml.dump(config, allow_unicode=True))

    return config


# ------------------------------------------------------------
# 2. 测试主流程
# ------------------------------------------------------------
def main():

    print("\n=======================================")
    print("   TEST: train_tft.py 整体流程测试")
    print("=======================================\n")

    # 1) 加载 & 修改 config
    config = make_test_config()

    # 2) load data
    df = get_raw_data(config)
    print(f"[OK] 数据加载成功，数据量 = {len(df)}")

    # 3) preprocess
    df, scaler = preprocess(df, config)
    print("[OK] 预处理成功")

    # 4) datasets
    training, validation = create_datasets(df, config)
    print("[OK] 数据集创建成功")
    print(f"训练样本数: {len(training)}  | 验证样本数: {len(validation)}")

    # 5) create model
    model = create_model(training, config)
    print("[OK] 模型创建成功")

    # 6) run training (1 epoch)
    print("\n[INFO] 开始快速训练测试（1 epoch）...")
    run_training(model, training, validation, config)

    print("\n=======================================")
    print("🎉 TEST SUCCESS: train_tft.py 工作正常！")
    print("=======================================\n")


if __name__ == "__main__":
    main()
