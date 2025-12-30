# ============================================================
# test_batch_format.py
# 用于检查 validation dataloader 返回格式
# ============================================================

from code.train.train_tft import (
    get_config,
    get_raw_data,
    preprocess,
    create_datasets,
)

def main():
    print("=== Testing dataloader batch format ===")

    config = get_config()
    df = get_raw_data(config)
    df, scaler = preprocess(df, config)
    training, validation = create_datasets(df, config)

    dl = validation.to_dataloader(train=False, batch_size=4, num_workers=0)

    batch = next(iter(dl))

    print("\n=== batch type:", type(batch))
    print("batch length:", len(batch))
    print("batch[0] type:", type(batch[0]))
    print("batch[1] type:", type(batch[1]))

    print("\n=== batch keys ===")
    for k, v in batch[0].items():
        print(k, type(v), getattr(v, "shape", None))

    print("\n=== target tensor ===")
    print(batch[1][0].shape)


if __name__ == "__main__":
    main()
