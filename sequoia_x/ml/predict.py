"""Alpha158 + LightGBM AI预测。

功能：
    1. 加载已经训练好的 LightGBM 模型
    2. 读取股票最新行情
    3. 生成 Alpha158 风格特征
    4. 预测未来 5 个交易日收益
    5. 对股票进行排序
    6. 输出 AI TOP10

注意：
    当前阶段用于验证完整 AI 预测链路。
"""

from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from sequoia_x.core.config import get_settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.ml.alpha158 import build_alpha158_features


logger = logging.getLogger(__name__)


MODEL_PATH = Path(
    "models/alpha158_lightgbm.txt"
)


def predict_top_stocks(
    engine: DataEngine,
    max_stocks: int = 100,
    top_k: int = 10,
) -> pd.DataFrame:
    """使用 LightGBM 预测股票未来收益。

    参数：
        engine:
            Sequoia-X 数据引擎。

        max_stocks:
            当前测试阶段最多预测多少只股票。
            默认100只。

        top_k:
            返回排名前多少只股票。
            默认10只。

    返回：
        包含股票代码和预测收益率的 DataFrame。
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"模型文件不存在：{MODEL_PATH}"
        )

    if max_stocks <= 0:
        raise ValueError(
            "max_stocks 必须大于 0"
        )

    if top_k <= 0:
        raise ValueError(
            "top_k 必须大于 0"
        )

    logger.info("=" * 60)
    logger.info("开始执行 Alpha158 + LightGBM AI预测")
    logger.info("=" * 60)

    # --------------------------------------------------
    # 加载 LightGBM 模型
    # --------------------------------------------------

    logger.info(
        f"加载模型：{MODEL_PATH}"
    )

    model = lgb.Booster(
        model_file=str(MODEL_PATH)
    )

    model_features = model.feature_name()

    logger.info(
        f"模型特征数量：{len(model_features)}"
    )

    # --------------------------------------------------
    # 获取股票列表
    # --------------------------------------------------

    symbols = engine.get_local_symbols()

    if not symbols:
        raise RuntimeError(
            "数据库中没有股票数据"
        )

    symbols = symbols[:max_stocks]

    logger.info(
        f"本次AI预测股票数量：{len(symbols)}"
    )

    results: list[dict] = []

    # --------------------------------------------------
    # 逐只股票生成最新特征
    # --------------------------------------------------

    for index, symbol in enumerate(
        symbols,
        start=1,
    ):
        try:
            df = engine.get_ohlcv(symbol)

            if df.empty:
                logger.warning(
                    f"[{symbol}] 没有行情数据"
                )
                continue

            if len(df) < 120:
                logger.warning(
                    f"[{symbol}] 历史数据不足"
                )
                continue

            features = build_alpha158_features(
                df
            )

            if features.empty:
                continue

            # 最新一个交易日
            latest = features.iloc[
                -1
            ].copy()

            # --------------------------------------------------
            # 检查模型需要的特征
            # --------------------------------------------------

            missing_features = [
                feature
                for feature in model_features
                if feature not in features.columns
            ]

            if missing_features:
                logger.warning(
                    f"[{symbol}] 缺少模型特征："
                    f"{missing_features}"
                )
                continue

            X = pd.DataFrame(
                [
                    latest[model_features]
                ]
            )

            X = X.replace(
                [
                    float("inf"),
                    float("-inf"),
                ],
                pd.NA,
            )

            X = X.apply(
                pd.to_numeric,
                errors="coerce",
            )

            if X.isna().any().any():
                logger.warning(
                    f"[{symbol}] 最新特征存在空值，跳过"
                )
                continue

            # --------------------------------------------------
            # LightGBM预测
            # --------------------------------------------------

            prediction = float(
                model.predict(X)[0]
            )

            results.append(
                {
                    "symbol": symbol,
                    "prediction": prediction,
                }
            )

            if index % 20 == 0:
                logger.info(
                    f"AI预测进度："
                    f"{index}/{len(symbols)}"
                )

        except Exception as exc:
            logger.warning(
                f"[{symbol}] AI预测失败：{exc}"
            )

    if not results:
        raise RuntimeError(
            "没有产生任何有效AI预测结果"
        )

    result_df = pd.DataFrame(
        results
    )

    result_df = result_df.sort_values(
        "prediction",
        ascending=False,
    ).reset_index(
        drop=True
    )

    result_df["rank"] = (
        result_df.index + 1
    )

    top_result = result_df.head(
        top_k
    ).copy()

    logger.info("=" * 60)
    logger.info("AI TOP10")
    logger.info("=" * 60)

    for _, row in top_result.iterrows():
        logger.info(
            f"{int(row['rank']):2d}. "
            f"{row['symbol']}  "
            f"预测5日收益："
            f"{row['prediction'] * 100:+.2f}%"
        )

    logger.info("=" * 60)

    return top_result


def main() -> None:
    """运行AI预测测试。"""

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

    engine = DataEngine(
        settings
    )

    result = predict_top_stocks(
        engine,
        max_stocks=500,
        top_k=10,
    )

    print()
    print("=" * 60)
    print("AI TOP10")
    print("=" * 60)

    for _, row in result.iterrows():
        print(
            f"{int(row['rank']):2d}. "
            f"{row['symbol']}    "
            f"预测5日收益："
            f"{row['prediction'] * 100:+.2f}%"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()
