# -*- coding: utf-8 -*-
"""
Alpha158 + LightGBM AI预测模块。

功能：
    1. 加载已经训练好的 LightGBM 模型
    2. 加载训练阶段保存的特征列表
    3. 读取股票最新行情
    4. 生成 Alpha158 风格特征
    5. 严格按照训练阶段的特征顺序进行预测
    6. 预测未来 5 个交易日收益
    7. 对股票进行排序
    8. 输出 AI TOP10
    9. 可保存完整预测结果

重要原则：
    - 训练和预测必须使用完全一致的特征
    - 不允许因为特征顺序不同导致模型输入错位
    - 不使用未来数据
    - 只使用最新一个已经存在的交易日进行预测
    - 本模块负责“预测”，严格历史回测由 backtest.py 单独负责
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from sequoia_x.core.config import get_settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.ml.alpha158 import build_alpha158_features


logger = logging.getLogger(__name__)


# ============================================================
# 文件路径
# ============================================================

MODEL_DIR = Path("models")

MODEL_PATH = (
    MODEL_DIR / "alpha158_lightgbm.txt"
)

FEATURE_PATH = (
    MODEL_DIR / "alpha158_features.json"
)

PREDICTION_PATH = (
    MODEL_DIR / "alpha158_predictions.csv"
)


# ============================================================
# 默认参数
# ============================================================

DEFAULT_MAX_STOCKS = 500
DEFAULT_TOP_K = 10
MIN_HISTORY = 120


# ============================================================
# 工具函数
# ============================================================


def _load_feature_names() -> list[str]:
    """
    加载训练阶段保存的特征名称。

    trainer.py 会在训练完成后生成：

        models/alpha158_features.json

    预测阶段必须使用完全相同的特征顺序。
    """

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            "训练阶段的特征文件不存在："
            f"{FEATURE_PATH}\n"
            "请先运行 trainer.py 完成模型训练。"
        )

    try:
        with FEATURE_PATH.open(
            "r",
            encoding="utf-8",
        ) as f:
            payload = json.load(f)

    except Exception as exc:
        raise RuntimeError(
            f"读取特征文件失败：{FEATURE_PATH}"
        ) from exc

    # trainer.py 保存的是：
    #
    # {
    #     "feature_columns": [...]
    # }
    #
    # 同时兼容直接保存 list 的情况。

    if isinstance(payload, dict):
        feature_names = payload.get(
            "feature_columns"
        )

        if feature_names is None:
            feature_names = payload.get(
                "features"
            )

    elif isinstance(payload, list):
        feature_names = payload

    else:
        feature_names = None

    if not isinstance(
        feature_names,
        list,
    ):
        raise ValueError(
            f"无法从 {FEATURE_PATH} 中读取特征列表"
        )

    feature_names = [
        str(feature)
        for feature in feature_names
        if str(feature).strip()
    ]

    if not feature_names:
        raise ValueError(
            "训练阶段保存的特征列表为空"
        )

    return feature_names


def _load_model() -> lgb.Booster:
    """
    加载 LightGBM 模型。
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "模型文件不存在："
            f"{MODEL_PATH}\n"
            "请先运行 trainer.py 完成模型训练。"
        )

    logger.info(
        f"加载LightGBM模型：{MODEL_PATH}"
    )

    model = lgb.Booster(
        model_file=str(MODEL_PATH)
    )

    return model


def _clean_prediction_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    清洗模型输入特征。

    处理：
        - inf
        - -inf
        - 非数值
        - NaN
    """

    result = data.copy()

    result = result.replace(
        [
            float("inf"),
            float("-inf"),
        ],
        pd.NA,
    )

    for column in result.columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    return result


def _get_latest_row(
    df: pd.DataFrame,
) -> tuple[pd.Series, str]:
    """
    获取股票最新一个有效交易日的数据。

    返回：
        latest_row
        latest_date
    """

    if df.empty:
        raise ValueError(
            "行情数据为空"
        )

    data = df.copy()

    # --------------------------------------------------------
    # 标准化日期
    # --------------------------------------------------------

    if "date" in data.columns:
        data["date"] = pd.to_datetime(
            data["date"],
            errors="coerce",
        )

        data = data.dropna(
            subset=["date"]
        )

        data = data.sort_values(
            "date"
        )

    if data.empty:
        raise ValueError(
            "没有有效交易日期"
        )

    latest_row = data.iloc[-1]

    if "date" in data.columns:
        latest_date = (
            pd.Timestamp(
                latest_row["date"]
            ).strftime("%Y-%m-%d")
        )
    else:
        latest_date = ""

    return latest_row, latest_date


# ============================================================
# 核心预测函数
# ============================================================


def predict_top_stocks(
    engine: DataEngine,
    max_stocks: int = DEFAULT_MAX_STOCKS,
    top_k: int = DEFAULT_TOP_K,
    save_result: bool = True,
) -> pd.DataFrame:
    """
    使用 Alpha158 + LightGBM 预测股票未来收益。

    参数：
        engine:
            Sequoia-X 数据引擎。

        max_stocks:
            最多预测多少只股票。

        top_k:
            返回排名前多少只股票。

        save_result:
            是否保存完整预测结果。

    返回：
        TOP K 股票 DataFrame。
    """

    # ========================================================
    # 参数检查
    # ========================================================

    if max_stocks <= 0:
        raise ValueError(
            "max_stocks 必须大于 0"
        )

    if top_k <= 0:
        raise ValueError(
            "top_k 必须大于 0"
        )

    # ========================================================
    # 开始
    # ========================================================

    logger.info("=" * 70)
    logger.info(
        "开始执行 Alpha158 + LightGBM AI预测"
    )
    logger.info("=" * 70)

    # ========================================================
    # 加载模型
    # ========================================================

    model = _load_model()

    model_features = list(
        model.feature_name()
    )

    logger.info(
        f"LightGBM模型特征数量："
        f"{len(model_features)}"
    )

    # ========================================================
    # 加载训练阶段保存的特征列表
    # ========================================================

    saved_features = _load_feature_names()

    logger.info(
        f"训练阶段特征数量："
        f"{len(saved_features)}"
    )

    # --------------------------------------------------------
    # 严格检查模型特征与训练特征是否一致
    # --------------------------------------------------------

    if model_features != saved_features:
        model_set = set(model_features)
        saved_set = set(saved_features)

        missing_in_model = sorted(
            saved_set - model_set
        )

        extra_in_model = sorted(
            model_set - saved_set
        )

        raise RuntimeError(
            "模型特征列表与训练阶段保存的特征列表不一致。\n"
            f"模型缺少特征：{missing_in_model}\n"
            f"模型额外特征：{extra_in_model}\n"
            "请重新运行 trainer.py 生成一致的模型和特征文件。"
        )

    # 最终预测必须严格使用模型中的顺序
    feature_columns = model_features

    # ========================================================
    # 获取股票列表
    # ========================================================

    symbols = engine.get_local_symbols()

    if not symbols:
        raise RuntimeError(
            "数据库中没有股票数据"
        )

    symbols = list(symbols)[
        :max_stocks
    ]

    logger.info(
        f"本次AI预测股票数量："
        f"{len(symbols)}"
    )

    # ========================================================
    # 逐只股票预测
    # ========================================================

    results: list[dict] = []

    for index, symbol in enumerate(
        symbols,
        start=1,
    ):

        try:
            # ------------------------------------------------
            # 读取历史行情
            # ------------------------------------------------

            df = engine.get_ohlcv(
                symbol
            )

            if df is None or df.empty:
                logger.warning(
                    f"[{symbol}] 没有行情数据"
                )
                continue

            if len(df) < MIN_HISTORY:
                logger.warning(
                    f"[{symbol}] "
                    f"历史数据不足"
                    f"（{len(df)} < {MIN_HISTORY}）"
                )
                continue

            # ------------------------------------------------
            # 获取最新交易日
            # ------------------------------------------------

            _, latest_date = _get_latest_row(
                df
            )

            # ------------------------------------------------
            # 构造 Alpha158 特征
            # ------------------------------------------------

            features = build_alpha158_features(
                df
            )

            if features is None:
                logger.warning(
                    f"[{symbol}] "
                    "Alpha158特征生成失败"
                )
                continue

            if features.empty:
                logger.warning(
                    f"[{symbol}] "
                    "Alpha158特征为空"
                )
                continue

            # ------------------------------------------------
            # 检查模型所需特征
            # ------------------------------------------------

            missing_features = [
                feature
                for feature in feature_columns
                if feature not in features.columns
            ]

            if missing_features:
                logger.warning(
                    f"[{symbol}] "
                    f"缺少模型特征："
                    f"{missing_features[:10]}"
                )

                if len(missing_features) > 10:
                    logger.warning(
                        f"[{symbol}] "
                        f"另外还有 "
                        f"{len(missing_features) - 10} "
                        "个缺失特征"
                    )

                continue

            # ------------------------------------------------
            # 最新一个交易日
            # ------------------------------------------------

            latest = (
                features.iloc[-1]
                .copy()
            )

            # ------------------------------------------------
            # 构造模型输入
            #
            # 注意：
            # 必须严格按照训练阶段的特征顺序。
            # ------------------------------------------------

            X = pd.DataFrame(
                [
                    latest[
                        feature_columns
                    ]
                ],
                columns=feature_columns,
            )

            # ------------------------------------------------
            # 数值清洗
            # ------------------------------------------------

            X = _clean_prediction_features(
                X
            )

            # ------------------------------------------------
            # 检查缺失值
            # ------------------------------------------------

            if X.isna().any().any():
                missing_input = [
                    column
                    for column in X.columns
                    if pd.isna(
                        X.iloc[0][column]
                    )
                ]

                logger.warning(
                    f"[{symbol}] "
                    "最新特征存在空值，跳过。"
                    f"缺失特征："
                    f"{missing_input[:10]}"
                )

                continue

            # ------------------------------------------------
            # LightGBM预测
            # ------------------------------------------------

            prediction = float(
                model.predict(X)[0]
            )

            # ------------------------------------------------
            # 获取最新价格
            # ------------------------------------------------

            latest_close = None

            if "close" in df.columns:
                latest_close = pd.to_numeric(
                    df["close"].iloc[-1],
                    errors="coerce",
                )

                if pd.isna(
                    latest_close
                ):
                    latest_close = None
                else:
                    latest_close = float(
                        latest_close
                    )

            # ------------------------------------------------
            # 保存结果
            # ------------------------------------------------

            results.append(
                {
                    "symbol": symbol,
                    "date": latest_date,
                    "close": latest_close,
                    "prediction": prediction,
                }
            )

            # ------------------------------------------------
            # 进度日志
            # ------------------------------------------------

            if (
                index % 20 == 0
                or index == len(symbols)
            ):
                logger.info(
                    "AI预测进度："
                    f"{index}/{len(symbols)}"
                )

        except Exception as exc:

            logger.warning(
                f"[{symbol}] "
                f"AI预测失败：{exc}"
            )

    # ========================================================
    # 检查结果
    # ========================================================

    if not results:
        raise RuntimeError(
            "没有产生任何有效AI预测结果"
        )

    # ========================================================
    # 创建结果 DataFrame
    # ========================================================

    result_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # 删除无效预测
    # --------------------------------------------------------

    result_df["prediction"] = pd.to_numeric(
        result_df["prediction"],
        errors="coerce",
    )

    result_df = result_df.dropna(
        subset=["prediction"]
    )

    if result_df.empty:
        raise RuntimeError(
            "所有AI预测结果均无效"
        )

    # --------------------------------------------------------
    # 清理 Inf
    # --------------------------------------------------------

    result_df = result_df.replace(
        [
            float("inf"),
            float("-inf"),
        ],
        pd.NA,
    )

    result_df = result_df.dropna(
        subset=["prediction"]
    )

    # ========================================================
    # 排序
    # ========================================================

    result_df = result_df.sort_values(
        by="prediction",
        ascending=False,
    ).reset_index(
        drop=True
    )

    # ========================================================
    # 生成排名
    # ========================================================

    result_df["rank"] = (
        result_df.index + 1
    )

    # ========================================================
    # 保存完整预测结果
    #
    # 注意：
    # 保存的是全部预测结果，而不是只有TOP10。
    #
    # 后面的严格回测、策略分析可以直接利用。
    # ========================================================

    if save_result:

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        result_df.to_csv(
            PREDICTION_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        logger.info(
            f"完整AI预测结果已保存："
            f"{PREDICTION_PATH}"
        )

    # ========================================================
    # TOP K
    # ========================================================

    actual_top_k = min(
        top_k,
        len(result_df),
    )

    top_result = (
        result_df
        .head(actual_top_k)
        .copy()
    )

    # ========================================================
    # 输出 TOP K
    # ========================================================

    logger.info("=" * 70)
    logger.info(
        f"AI TOP{actual_top_k}"
    )
    logger.info("=" * 70)

    for _, row in top_result.iterrows():

        prediction_pct = (
            float(row["prediction"])
            * 100
        )

        close_text = ""

        if pd.notna(
            row.get("close")
        ):
            close_text = (
                f"  收盘价："
                f"{float(row['close']):.2f}"
            )

        logger.info(
            f"{int(row['rank']):2d}. "
            f"{row['symbol']}  "
            f"预测5日收益："
            f"{prediction_pct:+.2f}%"
            f"{close_text}"
        )

    logger.info("=" * 70)

    return top_result


# ============================================================
# 主程序
# ============================================================


def main() -> None:
    """
    运行AI预测。
    """

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "[%(asctime)s] "
            "%(levelname)-8s "
            "%(name)s - "
            "%(message)s"
        ),
    )

    logger.info(
        "初始化Sequoia-X数据引擎"
    )

    settings = get_settings()

    engine = DataEngine(
        settings
    )

    result = predict_top_stocks(
        engine,
        max_stocks=DEFAULT_MAX_STOCKS,
        top_k=DEFAULT_TOP_K,
        save_result=True,
    )

    # ========================================================
    # 控制台输出
    # ========================================================

    print()
    print("=" * 70)
    print("Alpha158 + LightGBM AI TOP10")
    print("=" * 70)

    for _, row in result.iterrows():

        prediction_pct = (
            float(row["prediction"])
            * 100
        )

        close_text = ""

        if pd.notna(
            row.get("close")
        ):
            close_text = (
                f"    收盘价："
                f"{float(row['close']):.2f}"
            )

        print(
            f"{int(row['rank']):2d}. "
            f"{row['symbol']}    "
            f"预测5日收益："
            f"{prediction_pct:+.2f}%"
            f"{close_text}"
        )

    print("=" * 70)

    print(
        f"完整预测结果："
        f"{PREDICTION_PATH}"
    )


# ============================================================
# 程序入口
# ============================================================


if __name__ == "__main__":
    main()
