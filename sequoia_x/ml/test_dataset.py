"""测试 Alpha158 + LightGBM 训练数据集。"""

from __future__ import annotations

from sequoia_x.core.config import get_settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.ml.dataset import (
    build_training_dataset,
    get_feature_columns,
)


def main() -> None:
    """执行 Alpha158 数据测试。"""

    print("=" * 60)
    print("Sequoia-X Alpha158 数据测试")
    print("=" * 60)

    settings = get_settings()
    engine = DataEngine(settings)

    symbols = engine.get_local_symbols()

    print(f"数据库股票数量：{len(symbols)}")

    if not symbols:
        print("错误：数据库没有股票数据")
        raise SystemExit(1)

    # 为了第一次测试速度快，
    # 只测试第一只股票。
    symbol = symbols[0]

    print(f"测试股票：{symbol}")

    df = engine.get_ohlcv(symbol)

    print(f"原始数据行数：{len(df)}")

    if df.empty:
        print("错误：股票没有历史数据")
        raise SystemExit(1)

    dataset = build_training_dataset(
        df,
        horizon=5,
    )

    if dataset.empty:
        print("错误：没有生成训练数据")
        raise SystemExit(1)

    feature_columns = get_feature_columns(
        dataset
    )

    print("-" * 60)
    print(f"训练数据行数：{len(dataset)}")
    print(f"Alpha特征数量：{len(feature_columns)}")
    print("-" * 60)

    print("Alpha特征：")

    for feature in feature_columns:
        print(f"  - {feature}")

    print("-" * 60)

    target = dataset["target_return_5d"]

    print(
        "未来5日收益平均值："
        f"{target.mean():.4%}"
    )

    print(
        "未来5日收益中位数："
        f"{target.median():.4%}"
    )

    print(
        "未来5日收益最大值："
        f"{target.max():.4%}"
    )

    print(
        "未来5日收益最小值："
        f"{target.min():.4%}"
    )

    print("-" * 60)
    print("最近5条训练数据：")

    columns_to_show = [
        "date",
        "close",
        "target_return_5d",
    ]

    available_columns = [
        column
        for column in columns_to_show
        if column in dataset.columns
    ]

    print(
        dataset[available_columns]
        .tail(5)
        .to_string(index=False)
    )

    print("=" * 60)
    print("Alpha158 数据测试成功")
    print("=" * 60)


if __name__ == "__main__":
    main()
