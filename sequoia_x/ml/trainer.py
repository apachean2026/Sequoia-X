"""Alpha158 + LightGBM 模型训练器。

第一阶段：
    先使用少量股票验证完整训练流程。

后续阶段：
    扩展到全部股票，并进行模型优化。
"""

from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from sequoia_x.core.config import get_settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.ml.dataset import (
    build_training_dataset,
    prepare_lightgbm_data,
)


logger = logging.getLogger(__name__)


MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "alpha158_lightgbm.txt"


def train_model(
    engine: DataEngine,
    max_stocks: int = 20,
) -> dict:
    """训练 Alpha158 + LightGBM 模型。

    参数：
        engine:
            Sequoia-X 数据引擎。
        max_stocks:
            第一阶段最多训练多少只股票。

    返回：
        包含训练结果的字典。
    """

    logger.info("=" * 60)
    logger.info("开始训练 Alpha158 + LightGBM 模型")
    logger.info("=" * 60)

    symbols = engine.get_local_symbols()

    if not symbols:
        raise RuntimeError("数据库中没有股票数据")

    symbols = symbols[:max_stocks]

    logger.info(
        f"本次训练股票数量：{len(symbols)}"
    )

    datasets: list[pd.DataFrame] = []

    for index, symbol in enumerate(symbols, start=1):
        try:
            df = engine.get_ohlcv(symbol)

            if df.empty:
                logger.warning(
                    f"[{symbol}] 没有行情数据"
                )
                continue

            dataset = build_training_dataset(df)

            if dataset.empty:
                logger.warning(
                    f"[{symbol}] 训练数据不足"
                )
                continue

            dataset["symbol"] = symbol

            datasets.append(dataset)

            logger.info(
                f"[{index}/{len(symbols)}] "
                f"{symbol}：{len(dataset)} 条训练数据"
            )

        except Exception as exc:
            logger.warning(
                f"[{symbol}] 数据处理失败：{exc}"
            )

    if not datasets:
        raise RuntimeError(
            "没有生成任何有效训练数据"
        )

    data = pd.concat(
        datasets,
        ignore_index=True,
    )

    logger.info(
        f"训练数据总量：{len(data)}"
    )

    X, y, feature_columns = prepare_lightgbm_data(
        data
    )

    if X.empty or y.empty:
        raise RuntimeError(
            "LightGBM 训练数据为空"
        )

    logger.info(
        f"Alpha特征数量：{len(feature_columns)}"
    )

    logger.info(
        f"有效样本数量：{len(X)}"
    )

    # --------------------------------------------------
    # 时间序列切分
    # --------------------------------------------------
    #
    # 不随机打乱数据。
    # 使用前 80% 训练，后 20% 验证。
    # 这样可以避免未来数据泄漏。
    # --------------------------------------------------

    data_for_split = data.loc[X.index].copy()

    data_for_split["date"] = pd.to_datetime(
        data_for_split["date"],
        errors="coerce",
    )

    sort_index = (
        data_for_split["date"]
        .sort_values()
        .index
    )

    X = X.loc[sort_index]
    y = y.loc[sort_index]

    split_index = int(len(X) * 0.8)

    if split_index <= 0 or split_index >= len(X):
        raise RuntimeError(
            "训练集/验证集切分失败"
        )

    X_train = X.iloc[:split_index]
    X_valid = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_valid = y.iloc[split_index:]

    logger.info(
        f"训练集：{len(X_train)}"
    )

    logger.info(
        f"验证集：{len(X_valid)}"
    )

    # --------------------------------------------------
    # LightGBM
    # --------------------------------------------------

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    logger.info(
        "开始 LightGBM 训练..."
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_valid, y_valid),
        ],
        eval_metric="l2",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=30,
                verbose=False,
            )
        ],
    )

    # --------------------------------------------------
    # 验证集预测
    # --------------------------------------------------

    predictions = model.predict(
        X_valid
    )

    mse = float(
        (
            (predictions - y_valid) ** 2
        ).mean()
    )

    rmse = mse ** 0.5

    mae = float(
        (
            predictions - y_valid
        ).abs().mean()
    )

    logger.info("=" * 60)
    logger.info("LightGBM 训练完成")
    logger.info(f"RMSE：{rmse:.6f}")
    logger.info(f"MAE ：{mae:.6f}")
    logger.info(
        f"最佳迭代次数："
        f"{model.best_iteration_}"
    )
    logger.info("=" * 60)

    # --------------------------------------------------
    # 保存模型
    # --------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.booster_.save_model(
        str(MODEL_PATH)
    )

    logger.info(
        f"模型已保存：{MODEL_PATH}"
    )

    return {
        "stocks": len(symbols),
        "samples": len(X),
        "features": len(feature_columns),
        "train_samples": len(X_train),
        "valid_samples": len(X_valid),
        "rmse": rmse,
        "mae": mae,
        "best_iteration": model.best_iteration_,
        "model_path": str(MODEL_PATH),
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "[%(asctime)s] "
            "%(levelname)-8s "
            "%(name)s - "
            "%(message)s"
        ),
    )

    settings = get_settings()

    engine = DataEngine(settings)

    result = train_model(
        engine,
        max_stocks=20,
    )

    print()
    print("=" * 60)
    print("Alpha158 + LightGBM 训练成功")
    print("=" * 60)
    print(f"训练股票：{result['stocks']}")
    print(f"训练样本：{result['samples']}")
    print(f"Alpha特征：{result['features']}")
    print(f"训练集：{result['train_samples']}")
    print(f"验证集：{result['valid_samples']}")
    print(f"RMSE：{result['rmse']:.6f}")
    print(f"MAE：{result['mae']:.6f}")
    print(f"最佳迭代：{result['best_iteration']}")
    print(f"模型：{result['model_path']}")
    print("=" * 60)
