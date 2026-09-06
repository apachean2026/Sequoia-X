# -*- coding: utf-8 -*-
"""
Alpha158 + LightGBM 严格历史回测。

功能：
    1. 使用历史数据进行滚动训练
    2. 严格避免未来数据泄露
    3. 每隔固定交易日重新训练模型
    4. 对历史横截面股票进行预测
    5. 计算 IC
    6. 计算 Rank IC
    7. 计算 AI TOP10 胜率
    8. 计算 TOP10 平均未来5日收益
    9. 计算累计收益
    10. 计算最大回撤
    11. 计算年化收益
    12. 计算 Sharpe
    13. 保存逐轮回测结果
    14. 保存 TOP10 股票明细

严格回测原则：

    - 训练数据只能使用预测日期之前已经完成的历史数据
    - 训练标签对应的未来5个交易日不能跨越预测日期
    - 不随机打乱时间序列
    - 训练集必须早于验证集
    - 每轮回测重新训练模型
    - 预测阶段只使用预测日当时可获得的 Alpha158 特征
    - 不使用未来收盘价构造预测特征
    - 不使用当前已经训练好的模型进行历史回测

说明：

    当前默认：
        股票数量：500
        回测周期：最近250个交易日
        每20个交易日重新训练一次
        预测未来5个交易日收益
        每次选择TOP10

    这是一版研究/验证级回测。
    回测结果不代表未来实盘收益。
"""

from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import pandas as pd


from sequoia_x.core.config import get_settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.ml.alpha158 import build_alpha158_features
from sequoia_x.ml.dataset import get_feature_columns


logger = logging.getLogger(__name__)


# ============================================================
# 回测参数
# ============================================================

# 最多使用多少只股票
MAX_STOCKS = 500

# 最近多少个交易日作为回测区间
BACKTEST_DAYS = 250

# 每多少个交易日重新训练一次模型
RETRAIN_EVERY = 20

# 预测未来多少个交易日收益
HORIZON = 5

# 每次选择多少只股票
TOP_K = 10

# 最少历史数据
MIN_HISTORY = 180

# 最少训练样本
MIN_TRAIN_SAMPLES = 1000

# 训练集比例
TRAIN_RATIO = 0.80

# 输出文件
RESULT_PATH = Path(
    "backtest_results.csv"
)


# ============================================================
# LightGBM 参数
# ============================================================

MODEL_PARAMS = {
    "objective": "regression",

    # 与 trainer.py 保持基本一致
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,

    "min_child_samples": 30,

    "subsample": 0.8,
    "subsample_freq": 1,

    "colsample_bytree": 0.8,

    "reg_alpha": 0.1,
    "reg_lambda": 0.1,

    "random_state": 42,

    "n_jobs": -1,
    "verbosity": -1,
}


# ============================================================
# 基础工具
# ============================================================


def calculate_ic(
    prediction: pd.Series,
    actual: pd.Series,
) -> float:
    """
    计算 Pearson IC。

    IC 衡量：
        模型预测值
        与
        实际未来收益

    之间的线性相关性。
    """

    data = pd.DataFrame(
        {
            "prediction": prediction,
            "actual": actual,
        }
    )

    data = data.replace(
        [
            float("inf"),
            float("-inf"),
        ],
        pd.NA,
    )

    data = data.dropna()

    if len(data) < 3:
        return float("nan")

    if data["prediction"].nunique() <= 1:
        return float("nan")

    if data["actual"].nunique() <= 1:
        return float("nan")

    return float(
        data["prediction"].corr(
            data["actual"],
            method="pearson",
        )
    )


def calculate_rank_ic(
    prediction: pd.Series,
    actual: pd.Series,
) -> float:
    """
    计算 Spearman Rank IC。

    Rank IC 更关注：
        模型排序能力
    而不是预测绝对收益值的精确程度。
    """

    data = pd.DataFrame(
        {
            "prediction": prediction,
            "actual": actual,
        }
    )

    data = data.replace(
        [
            float("inf"),
            float("-inf"),
        ],
        pd.NA,
    )

    data = data.dropna()

    if len(data) < 3:
        return float("nan")

    if data["prediction"].nunique() <= 1:
        return float("nan")

    if data["actual"].nunique() <= 1:
        return float("nan")

    return float(
        data["prediction"].corr(
            data["actual"],
            method="spearman",
        )
    )


def calculate_max_drawdown(
    returns: pd.Series,
) -> float:
    """
    计算最大回撤。
    """

    returns = pd.to_numeric(
        returns,
        errors="coerce",
    ).dropna()

    if returns.empty:
        return float("nan")

    equity = (
        1.0 + returns
    ).cumprod()

    running_max = equity.cummax()

    drawdown = (
        equity / running_max
        - 1.0
    )

    return float(
        drawdown.min()
    )


def calculate_sharpe(
    returns: pd.Series,
) -> float:
    """
    计算基于回测轮次的 Sharpe。

    注意：
        当前回测每20个交易日进行一次组合调整，
        因此这里按照约20个交易日作为一个周期进行年化。

        sqrt(252 / RETRAIN_EVERY)
    """

    returns = pd.to_numeric(
        returns,
        errors="coerce",
    ).dropna()

    if len(returns) < 2:
        return float("nan")

    std = float(
        returns.std(ddof=1)
    )

    if std <= 0:
        return float("nan")

    mean_return = float(
        returns.mean()
    )

    annualization = (
        252.0 / RETRAIN_EVERY
    ) ** 0.5

    return float(
        mean_return
        / std
        * annualization
    )


def calculate_annualized_return(
    cumulative_return: float,
    backtest_days: int,
) -> float:
    """
    根据实际回测交易日数计算年化收益。
    """

    if backtest_days <= 0:
        return float("nan")

    equity = 1.0 + cumulative_return

    if equity <= 0:
        return float("nan")

    years = (
        backtest_days / 252.0
    )

    if years <= 0:
        return float("nan")

    return float(
        equity ** (1.0 / years)
        - 1.0
    )


# ============================================================
# 构建股票历史特征
# ============================================================


def build_stock_features(
    engine: DataEngine,
    symbols: list[str],
) -> dict[str, pd.DataFrame]:
    """
    一次性构建所有股票的 Alpha158 历史特征。

    同时构造：

        future_return

    和：

        future_date

    其中：

        future_return =
            未来5个交易日收盘价 / 当前收盘价 - 1

        future_date =
            未来第5个交易日日期

    future_date 非常重要。

    后面的严格回测会使用：

        future_date < prediction_date

    来确保训练标签在预测时已经完全结束。
    """

    logger.info("=" * 70)
    logger.info(
        "开始构建历史 Alpha158 特征"
    )
    logger.info("=" * 70)

    stock_data: dict[
        str,
        pd.DataFrame,
    ] = {}

    total = len(symbols)

    for index, symbol in enumerate(
        symbols,
        start=1,
    ):

        try:
            df = engine.get_ohlcv(
                symbol
            )

            if df is None or df.empty:
                continue

            if len(df) < MIN_HISTORY:
                continue

            # ------------------------------------------------
            # 标准化日期
            # ------------------------------------------------

            df = df.copy()

            if "date" not in df.columns:
                logger.warning(
                    f"[{symbol}] 缺少date字段"
                )
                continue

            df["date"] = pd.to_datetime(
                df["date"],
                errors="coerce",
            )

            df = df.dropna(
                subset=["date"]
            )

            df = df.sort_values(
                "date"
            )

            df = df.drop_duplicates(
                subset=["date"],
                keep="last",
            )

            if len(df) < MIN_HISTORY:
                continue

            # ------------------------------------------------
            # Alpha158 特征
            # ------------------------------------------------

            features = (
                build_alpha158_features(
                    df
                )
            )

            if features is None:
                continue

            if features.empty:
                continue

            features = features.copy()

            # ------------------------------------------------
            # 日期标准化
            # ------------------------------------------------

            if "date" not in features.columns:
                logger.warning(
                    f"[{symbol}] "
                    "Alpha158结果缺少date字段"
                )
                continue

            features["date"] = pd.to_datetime(
                features["date"],
                errors="coerce",
            )

            features = features.dropna(
                subset=["date"]
            )

            features = features.sort_values(
                "date"
            )

            features = features.drop_duplicates(
                subset=["date"],
                keep="last",
            )

            # ------------------------------------------------
            # 检查close
            # ------------------------------------------------

            if "close" not in features.columns:
                logger.warning(
                    f"[{symbol}] "
                    "Alpha158结果缺少close字段"
                )
                continue

            features["close"] = pd.to_numeric(
                features["close"],
                errors="coerce",
            )

            features = features.dropna(
                subset=["close"]
            )

            # ------------------------------------------------
            # 构造未来5个交易日收益
            # ------------------------------------------------

            features[
                "future_date"
            ] = features[
                "date"
            ].shift(
                -HORIZON
            )

            features[
                "future_return"
            ] = (
                features[
                    "close"
                ].shift(
                    -HORIZON
                )
                / features[
                    "close"
                ]
                - 1.0
            )

            # ------------------------------------------------
            # 清理无穷值
            # ------------------------------------------------

            features = features.replace(
                [
                    float("inf"),
                    float("-inf"),
                ],
                pd.NA,
            )

            features = features.reset_index(
                drop=True
            )

            if len(features) < MIN_HISTORY:
                continue

            stock_data[
                symbol
            ] = features

            if (
                index % 25 == 0
                or index == total
            ):
                logger.info(
                    "特征构建进度："
                    f"{index}/{total}，"
                    f"有效股票："
                    f"{len(stock_data)}"
                )

        except Exception as exc:

            logger.warning(
                f"[{symbol}] "
                f"特征构建失败：{exc}"
            )

    logger.info(
        f"历史特征构建完成，"
        f"有效股票数量："
        f"{len(stock_data)}"
    )

    return stock_data


# ============================================================
# 获取统一特征列
# ============================================================


def get_backtest_feature_names(
    stock_data: dict[str, pd.DataFrame],
) -> list[str]:
    """
    获取回测统一使用的特征列。

    采用：
        所有股票共同存在的特征

    避免某一只股票特征缺失造成训练/预测列不一致。
    """

    if not stock_data:
        return []

    common_columns: set[str] | None = None

    for df in stock_data.values():

        columns = set(
            get_feature_columns(
                df
            )
        )

        columns.discard(
            "future_return"
        )

        columns.discard(
            "future_date"
        )

        columns.discard(
            "date"
        )

        columns.discard(
            "symbol"
        )

        if common_columns is None:
            common_columns = columns
        else:
            common_columns &= columns

    if not common_columns:
        return []

    feature_columns = sorted(
        common_columns
    )

    # --------------------------------------------------------
    # 进一步删除明显无效特征
    # --------------------------------------------------------

    valid_features: list[str] = []

    for feature in feature_columns:

        has_valid_value = False
        has_variation = False

        for df in stock_data.values():

            if feature not in df.columns:
                continue

            series = pd.to_numeric(
                df[feature],
                errors="coerce",
            )

            series = series.replace(
                [
                    float("inf"),
                    float("-inf"),
                ],
                pd.NA,
            )

            series = series.dropna()

            if series.empty:
                continue

            has_valid_value = True

            if series.nunique() > 1:
                has_variation = True

            if has_variation:
                break

        if (
            has_valid_value
            and has_variation
        ):
            valid_features.append(
                feature
            )

    return valid_features


# ============================================================
# 构建严格训练数据
# ============================================================


def build_training_data(
    stock_data: dict[str, pd.DataFrame],
    prediction_date: pd.Timestamp,
    feature_columns: list[str],
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
]:
    """
    构造某一个预测日期之前的严格训练数据。

    最重要的规则：

        future_date < prediction_date

    即：

        训练样本的未来5日收益
        必须在预测日之前已经完全实现。

    举例：

        预测日：
            2026-01-20

        如果某训练样本：
            sample_date = 2026-01-10
            future_date = 2026-01-17

        可以使用。

        如果：
            sample_date = 2026-01-15
            future_date = 2026-01-22

        不可以使用。

    这样可以避免未来标签泄漏。
    """

    datasets: list[pd.DataFrame] = []

    for symbol, df in stock_data.items():

        if df.empty:
            continue

        data = df.copy()

        # ----------------------------------------------------
        # 严格限制：
        # 标签结束日期必须早于预测日期
        # ----------------------------------------------------

        data = data[
            data["future_date"]
            < prediction_date
        ].copy()

        if data.empty:
            continue

        # ----------------------------------------------------
        # 必须有有效标签
        # ----------------------------------------------------

        data = data.dropna(
            subset=[
                "future_return",
                "future_date",
            ]
        )

        if data.empty:
            continue

        # ----------------------------------------------------
        # 检查特征
        # ----------------------------------------------------

        required_columns = (
            feature_columns
            + [
                "future_return",
                "future_date",
                "date",
            ]
        )

        missing = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing:
            continue

        # ----------------------------------------------------
        # 数值化
        # ----------------------------------------------------

        for feature in feature_columns:
            data[feature] = pd.to_numeric(
                data[feature],
                errors="coerce",
            )

        data[
            "future_return"
        ] = pd.to_numeric(
            data["future_return"],
            errors="coerce",
        )

        # ----------------------------------------------------
        # 清理 Inf
        # ----------------------------------------------------

        data = data.replace(
            [
                float("inf"),
                float("-inf"),
            ],
            pd.NA,
        )

        # ----------------------------------------------------
        # 严格要求：
        # 所有模型输入特征都有效
        # ----------------------------------------------------

        data = data.dropna(
            subset=(
                feature_columns
                + [
                    "future_return"
                ]
            )
        )

        if data.empty:
            continue

        data["symbol"] = symbol

        datasets.append(
            data[
                feature_columns
                + [
                    "future_return",
                    "date",
                    "future_date",
                    "symbol",
                ]
            ]
        )

    if not datasets:
        return (
            pd.DataFrame(
                columns=feature_columns
            ),
            pd.Series(
                dtype=float
            ),
            pd.DataFrame(),
        )

    training_data = pd.concat(
        datasets,
        ignore_index=True,
    )

    # ========================================================
    # 极其重要：
    # 先按照日期排序
    # ========================================================

    training_data[
        "date"
    ] = pd.to_datetime(
        training_data["date"],
        errors="coerce",
    )

    training_data = training_data.sort_values(
        [
            "date",
            "symbol",
        ]
    ).reset_index(
        drop=True
    )

    # ========================================================
    # X / y
    # ========================================================

    X = training_data[
        feature_columns
    ].copy()

    y = pd.to_numeric(
        training_data[
            "future_return"
        ],
        errors="coerce",
    )

    # --------------------------------------------------------
    # 最终有效样本
    # --------------------------------------------------------

    valid_mask = (
        y.notna()
        & X.notna().all(axis=1)
    )

    X = X.loc[
        valid_mask
    ].reset_index(
        drop=True
    )

    y = y.loc[
        valid_mask
    ].reset_index(
        drop=True
    )

    modeling_data = (
        training_data.loc[
            valid_mask
        ]
        .reset_index(
            drop=True
        )
    )

    return (
        X,
        y,
        modeling_data,
    )


# ============================================================
# 训练回测模型
# ============================================================


def train_backtest_model(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[
    lgb.LGBMRegressor,
    dict,
]:
    """
    使用严格时间序列方式训练模型。

    不能随机切分。

    训练：
        前80%

    验证：
        后20%

    并且：
        验证集一定发生在训练集之后。
    """

    if len(X) < MIN_TRAIN_SAMPLES:
        raise RuntimeError(
            f"训练样本过少："
            f"{len(X)}，"
            f"至少需要："
            f"{MIN_TRAIN_SAMPLES}"
        )

    # --------------------------------------------------------
    # 时间序列切分
    # --------------------------------------------------------

    split_index = int(
        len(X)
        * TRAIN_RATIO
    )

    if (
        split_index <= 0
        or split_index >= len(X)
    ):
        raise RuntimeError(
            "训练/验证集切分失败"
        )

    X_train = X.iloc[
        :split_index
    ].copy()

    X_valid = X.iloc[
        split_index:
    ].copy()

    y_train = y.iloc[
        :split_index
    ].copy()

    y_valid = y.iloc[
        split_index:
    ].copy()

    # --------------------------------------------------------
    # LightGBM
    # --------------------------------------------------------

    model = lgb.LGBMRegressor(
        **MODEL_PARAMS
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_valid,
                y_valid,
            )
        ],
        eval_metric="l2",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50,
                verbose=False,
            ),
            lgb.log_evaluation(
                period=0
            ),
        ],
    )

    # --------------------------------------------------------
    # 验证集指标
    # --------------------------------------------------------

    valid_prediction = model.predict(
        X_valid,
        num_iteration=(
            model.best_iteration_
            if model.best_iteration_
            else None
        ),
    )

    valid_prediction = pd.Series(
        valid_prediction
    )

    valid_actual = (
        y_valid
        .reset_index(drop=True)
    )

    valid_ic = calculate_ic(
        valid_prediction,
        valid_actual,
    )

    valid_rank_ic = calculate_rank_ic(
        valid_prediction,
        valid_actual,
    )

    valid_mae = float(
        (
            valid_prediction
            - valid_actual
        )
        .abs()
        .mean()
    )

    valid_rmse = float(
        (
            (
                valid_prediction
                - valid_actual
            )
            ** 2
        ).mean()
        ** 0.5
    )

    metrics = {
        "valid_ic": valid_ic,
        "valid_rank_ic": valid_rank_ic,
        "valid_mae": valid_mae,
        "valid_rmse": valid_rmse,
        "best_iteration": (
            model.best_iteration_
            if model.best_iteration_
            else MODEL_PARAMS[
                "n_estimators"
            ]
        ),
    }

    return (
        model,
        metrics,
    )


# ============================================================
# 单个交易日横截面预测
# ============================================================


def predict_one_date(
    model: lgb.LGBMRegressor,
    stock_data: dict[str, pd.DataFrame],
    prediction_date: pd.Timestamp,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    对某一个交易日进行横截面预测。

    注意：

        这里的 future_return
        只用于回测评价。

        模型输入 X
        只使用 prediction_date 当天
        已经存在的特征。

    因此：
        prediction
        与
        actual_return

    是严格分离的。
    """

    rows: list[dict] = []

    for symbol, df in stock_data.items():

        try:
            current = df[
                df["date"]
                == prediction_date
            ]

            if current.empty:
                continue

            row = current.iloc[
                -1
            ]

            # ------------------------------------------------
            # 模型输入
            # ------------------------------------------------

            X = pd.DataFrame(
                [
                    row[
                        feature_columns
                    ]
                ],
                columns=feature_columns,
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
                continue

            # ------------------------------------------------
            # 预测
            # ------------------------------------------------

            prediction = float(
                model.predict(
                    X,
                    num_iteration=(
                        model.best_iteration_
                        if model.best_iteration_
                        else None
                    ),
                )[0]
            )

            # ------------------------------------------------
            # 实际未来5日收益
            # ------------------------------------------------

            actual_return = pd.to_numeric(
                row[
                    "future_return"
                ],
                errors="coerce",
            )

            if pd.isna(
                actual_return
            ):
                continue

            actual_return = float(
                actual_return
            )

            rows.append(
                {
                    "symbol": symbol,
                    "date": prediction_date,
                    "prediction": prediction,
                    "actual_return": actual_return,
                }
            )

        except Exception as exc:

            logger.debug(
                f"[{symbol}] "
                f"预测失败：{exc}"
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "date",
                "prediction",
                "actual_return",
            ]
        )

    result = pd.DataFrame(
        rows
    )

    result = result.replace(
        [
            float("inf"),
            float("-inf"),
        ],
        pd.NA,
    )

    result = result.dropna(
        subset=[
            "prediction",
            "actual_return",
        ]
    )

    return result


# ============================================================
# 主回测函数
# ============================================================


def run_backtest(
    engine: DataEngine,
    max_stocks: int = MAX_STOCKS,
    backtest_days: int = BACKTEST_DAYS,
    retrain_every: int = RETRAIN_EVERY,
    top_k: int = TOP_K,
) -> dict:
    """
    执行严格历史滚动回测。
    """

    # ========================================================
    # 参数检查
    # ========================================================

    if max_stocks <= 0:
        raise ValueError(
            "max_stocks 必须大于0"
        )

    if backtest_days <= 0:
        raise ValueError(
            "backtest_days 必须大于0"
        )

    if retrain_every <= 0:
        raise ValueError(
            "retrain_every 必须大于0"
        )

    if top_k <= 0:
        raise ValueError(
            "top_k 必须大于0"
        )

    # ========================================================
    # 开始
    # ========================================================

    logger.info("=" * 70)
    logger.info(
        "开始 Alpha158 + LightGBM 严格历史回测"
    )
    logger.info("=" * 70)

    logger.info(
        f"股票数量上限：{max_stocks}"
    )

    logger.info(
        f"回测交易日：{backtest_days}"
    )

    logger.info(
        f"重新训练周期："
        f"{retrain_every}个交易日"
    )

    logger.info(
        f"预测周期："
        f"{HORIZON}个交易日"
    )

    logger.info(
        f"每轮TOP：{top_k}"
    )

    # ========================================================
    # 1. 获取股票
    # ========================================================

    symbols = engine.get_local_symbols()

    if not symbols:
        raise RuntimeError(
            "数据库中没有股票数据"
        )

    symbols = list(
        symbols
    )[:max_stocks]

    logger.info(
        f"实际使用股票："
        f"{len(symbols)}"
    )

    # ========================================================
    # 2. 构建全部历史特征
    # ========================================================

    stock_data = build_stock_features(
        engine,
        symbols,
    )

    if not stock_data:
        raise RuntimeError(
            "没有生成有效历史特征"
        )

    # ========================================================
    # 3. 获取统一特征
    # ========================================================

    feature_columns = (
        get_backtest_feature_names(
            stock_data
        )
    )

    if not feature_columns:
        raise RuntimeError(
            "没有找到有效Alpha特征"
        )

    logger.info(
        f"统一Alpha特征数量："
        f"{len(feature_columns)}"
    )

    # ========================================================
    # 4. 获取所有交易日
    # ========================================================

    all_dates_set: set[
        pd.Timestamp
    ] = set()

    for df in stock_data.values():

        dates = (
            pd.to_datetime(
                df["date"],
                errors="coerce",
            )
            .dropna()
            .tolist()
        )

        all_dates_set.update(
            dates
        )

    if not all_dates_set:
        raise RuntimeError(
            "没有找到有效交易日期"
        )

    all_dates = sorted(
        all_dates_set
    )

    # ========================================================
    # 5. 检查历史长度
    # ========================================================

    if len(all_dates) <= (
        backtest_days
        + MIN_HISTORY
    ):
        raise RuntimeError(
            "历史交易日不足，"
            "无法完成严格回测。"
            f"当前：{len(all_dates)}"
        )

    # ========================================================
    # 6. 选择回测区间
    # ========================================================

    backtest_dates = all_dates[
        -backtest_days:
    ]

    if len(backtest_dates) < 2:
        raise RuntimeError(
            "有效回测日期不足"
        )

    logger.info(
        f"回测开始："
        f"{backtest_dates[0].date()}"
    )

    logger.info(
        f"回测结束："
        f"{backtest_dates[-1].date()}"
    )

    logger.info(
        f"回测交易日数量："
        f"{len(backtest_dates)}"
    )

    # ========================================================
    # 7. 选择预测日期
    # ========================================================

    prediction_dates = (
        backtest_dates[
            ::retrain_every
        ]
    )

    logger.info(
        f"实际预测轮次："
        f"{len(prediction_dates)}"
    )

    # ========================================================
    # 8. 开始滚动回测
    # ========================================================

    results: list[dict] = []

    total_rounds = len(
        prediction_dates
    )

    for round_index, prediction_date in enumerate(
        prediction_dates,
        start=1,
    ):

        logger.info("")
        logger.info("=" * 70)
        logger.info(
            f"回测轮次："
            f"{round_index}/{total_rounds}"
        )

        logger.info(
            f"预测日期："
            f"{prediction_date.date()}"
        )

        # ====================================================
        # 训练数据
        # ====================================================

        (
            X_train,
            y_train,
            modeling_data,
        ) = build_training_data(
            stock_data,
            prediction_date,
            feature_columns,
        )

        if X_train.empty:
            logger.warning(
                "训练数据为空，跳过本轮"
            )
            continue

        if len(X_train) < MIN_TRAIN_SAMPLES:
            logger.warning(
                f"训练样本不足："
                f"{len(X_train)}"
            )
            continue

        # ====================================================
        # 训练数据时间范围
        # ====================================================

        train_start_date = (
            modeling_data[
                "date"
            ].min()
        )

        train_end_date = (
            modeling_data[
                "date"
            ].max()
        )

        label_end_date = (
            modeling_data[
                "future_date"
            ].max()
        )

        logger.info(
            f"训练样本："
            f"{len(X_train)}"
        )

        logger.info(
            f"训练开始日期："
            f"{train_start_date.date()}"
        )

        logger.info(
            f"训练最后样本日期："
            f"{train_end_date.date()}"
        )

        logger.info(
            f"训练最后标签日期："
            f"{label_end_date.date()}"
        )

        logger.info(
            f"预测日期："
            f"{prediction_date.date()}"
        )

        # ----------------------------------------------------
        # 最终安全检查
        # ----------------------------------------------------

        if label_end_date >= prediction_date:
            raise RuntimeError(
                "检测到潜在未来数据泄漏："
                f"训练标签结束日期"
                f"{label_end_date.date()} "
                f">= 预测日期"
                f"{prediction_date.date()}"
            )

        # ====================================================
        # 训练模型
        # ====================================================

        try:
            (
                model,
                validation_metrics,
            ) = train_backtest_model(
                X_train,
                y_train,
            )

        except Exception as exc:

            logger.warning(
                f"模型训练失败："
                f"{exc}"
            )

            continue

        logger.info(
            f"验证集RMSE："
            f"{validation_metrics['valid_rmse']:.6f}"
        )

        logger.info(
            f"验证集MAE："
            f"{validation_metrics['valid_mae']:.6f}"
        )

        logger.info(
            f"验证集IC："
            f"{validation_metrics['valid_ic']:.6f}"
        )

        logger.info(
            f"验证集Rank IC："
            f"{validation_metrics['valid_rank_ic']:.6f}"
        )

        logger.info(
            f"最佳迭代："
            f"{validation_metrics['best_iteration']}"
        )

        # ====================================================
        # 预测
        # ====================================================

        prediction_df = (
            predict_one_date(
                model,
                stock_data,
                prediction_date,
                feature_columns,
            )
        )

        if prediction_df.empty:
            logger.warning(
                "本轮没有有效预测"
            )
            continue

        logger.info(
            f"有效预测股票："
            f"{len(prediction_df)}"
        )

        # ====================================================
        # IC
        # ====================================================

        ic = calculate_ic(
            prediction_df[
                "prediction"
            ],
            prediction_df[
                "actual_return"
            ],
        )

        # ====================================================
        # Rank IC
        # ====================================================

        rank_ic = calculate_rank_ic(
            prediction_df[
                "prediction"
            ],
            prediction_df[
                "actual_return"
            ],
        )

        # ====================================================
        # TOP K
        # ====================================================

        top_result = (
            prediction_df
            .sort_values(
                "prediction",
                ascending=False,
            )
            .head(top_k)
            .copy()
        )

        if top_result.empty:
            continue

        # ====================================================
        # TOP K 平均收益
        # ====================================================

        top_return = float(
            top_result[
                "actual_return"
            ].mean()
        )

        # ====================================================
        # TOP K 胜率
        # ====================================================

        top_win_rate = float(
            (
                top_result[
                    "actual_return"
                ]
                > 0
            ).mean()
        )

        # ====================================================
        # TOP K 股票
        # ====================================================

        top_symbols = ",".join(
            top_result[
                "symbol"
            ].astype(str)
            .tolist()
        )

        # ====================================================
        # TOP K 最佳/最差
        # ====================================================

        top_best_return = float(
            top_result[
                "actual_return"
            ].max()
        )

        top_worst_return = float(
            top_result[
                "actual_return"
            ].min()
        )

        # ====================================================
        # 保存本轮
        # ====================================================

        results.append(
            {
                "date": prediction_date,

                "train_start_date":
                    train_start_date,

                "train_end_date":
                    train_end_date,

                "label_end_date":
                    label_end_date,

                "prediction_stocks":
                    len(prediction_df),

                "top10_count":
                    len(top_result),

                "ic":
                    ic,

                "rank_ic":
                    rank_ic,

                "validation_ic":
                    validation_metrics[
                        "valid_ic"
                    ],

                "validation_rank_ic":
                    validation_metrics[
                        "valid_rank_ic"
                    ],

                "validation_rmse":
                    validation_metrics[
                        "valid_rmse"
                    ],

                "validation_mae":
                    validation_metrics[
                        "valid_mae"
                    ],

                "best_iteration":
                    validation_metrics[
                        "best_iteration"
                    ],

                "top10_win_rate":
                    top_win_rate,

                "top10_avg_return":
                    top_return,

                "top10_best_return":
                    top_best_return,

                "top10_worst_return":
                    top_worst_return,

                "top10_symbols":
                    top_symbols,
            }
        )

        # ====================================================
        # 日志
        # ====================================================

        logger.info(
            f"IC："
            f"{ic:.6f}"
        )

        logger.info(
            f"Rank IC："
            f"{rank_ic:.6f}"
        )

        logger.info(
            f"TOP10胜率："
            f"{top_win_rate * 100:.2f}%"
        )

        logger.info(
            f"TOP10平均5日收益："
            f"{top_return * 100:+.2f}%"
        )

        logger.info(
            f"TOP10最佳收益："
            f"{top_best_return * 100:+.2f}%"
        )

        logger.info(
            f"TOP10最差收益："
            f"{top_worst_return * 100:+.2f}%"
        )

        logger.info(
            f"TOP10："
            f"{top_symbols}"
        )

    # ========================================================
    # 检查最终结果
    # ========================================================

    if not results:
        raise RuntimeError(
            "没有产生有效回测结果"
        )

    result_df = pd.DataFrame(
        results
    )

    # ========================================================
    # 日期排序
    # ========================================================

    result_df["date"] = pd.to_datetime(
        result_df["date"],
        errors="coerce",
    )

    result_df = result_df.dropna(
        subset=["date"]
    )

    result_df = result_df.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    # ========================================================
    # 累计收益
    # ========================================================

    result_df[
        "equity"
    ] = (
        1.0
        + result_df[
            "top10_avg_return"
        ]
    ).cumprod()

    result_df[
        "cumulative_return"
    ] = (
        result_df[
            "equity"
        ]
        - 1.0
    )

    # ========================================================
    # 运行最大值
    # ========================================================

    result_df[
        "running_max"
    ] = result_df[
        "equity"
    ].cummax()

    # ========================================================
    # 回撤
    # ========================================================

    result_df[
        "drawdown"
    ] = (
        result_df[
            "equity"
        ]
        / result_df[
            "running_max"
        ]
        - 1.0
    )

    # ========================================================
    # 汇总指标
    # ========================================================

    average_ic = float(
        result_df[
            "ic"
        ].mean()
    )

    average_rank_ic = float(
        result_df[
            "rank_ic"
        ].mean()
    )

    average_top10_win_rate = float(
        result_df[
            "top10_win_rate"
        ].mean()
    )

    average_top10_return = float(
        result_df[
            "top10_avg_return"
        ].mean()
    )

    cumulative_return = float(
        result_df[
            "cumulative_return"
        ].iloc[-1]
    )

    max_drawdown = float(
        result_df[
            "drawdown"
        ].min()
    )

    sharpe = calculate_sharpe(
        result_df[
            "top10_avg_return"
        ]
    )

    annualized_return = (
        calculate_annualized_return(
            cumulative_return,
            len(backtest_dates),
        )
    )

    profitable_rounds = int(
        (
            result_df[
                "top10_avg_return"
            ]
            > 0
        ).sum()
    )

    losing_rounds = int(
        (
            result_df[
                "top10_avg_return"
            ]
            < 0
        ).sum()
    )

    flat_rounds = int(
        (
            result_df[
                "top10_avg_return"
            ]
            == 0
        ).sum()
    )

    # ========================================================
    # 平均验证指标
    # ========================================================

    average_validation_ic = float(
        result_df[
            "validation_ic"
        ].mean()
    )

    average_validation_rank_ic = float(
        result_df[
            "validation_rank_ic"
        ].mean()
    )

    average_validation_rmse = float(
        result_df[
            "validation_rmse"
        ].mean()
    )

    average_validation_mae = float(
        result_df[
            "validation_mae"
        ].mean()
    )

    # ========================================================
    # 保存CSV
    # ========================================================

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 日期转字符串
    for column in [
        "date",
        "train_start_date",
        "train_end_date",
        "label_end_date",
    ]:
        if column in result_df.columns:
            result_df[
                column
            ] = pd.to_datetime(
                result_df[
                    column
                ],
                errors="coerce",
            ).dt.strftime(
                "%Y-%m-%d"
            )

    result_df.to_csv(
        RESULT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 输出最终结果
    # ========================================================

    logger.info("")
    logger.info("=" * 70)
    logger.info(
        "Alpha158 + LightGBM 严格历史回测完成"
    )
    logger.info("=" * 70)

    logger.info(
        f"训练股票数量："
        f"{len(symbols)}"
    )

    logger.info(
        f"有效股票数量："
        f"{len(stock_data)}"
    )

    logger.info(
        f"Alpha特征数量："
        f"{len(feature_columns)}"
    )

    logger.info(
        f"回测交易日："
        f"{len(backtest_dates)}"
    )

    logger.info(
        f"实际回测轮次："
        f"{len(result_df)}"
    )

    logger.info(
        f"平均IC："
        f"{average_ic:.6f}"
    )

    logger.info(
        f"平均Rank IC："
        f"{average_rank_ic:.6f}"
    )

    logger.info(
        f"验证集平均IC："
        f"{average_validation_ic:.6f}"
    )

    logger.info(
        f"验证集平均Rank IC："
        f"{average_validation_rank_ic:.6f}"
    )

    logger.info(
        f"验证集平均RMSE："
        f"{average_validation_rmse:.6f}"
    )

    logger.info(
        f"验证集平均MAE："
        f"{average_validation_mae:.6f}"
    )

    logger.info(
        f"TOP10平均胜率："
        f"{average_top10_win_rate * 100:.2f}%"
    )

    logger.info(
        f"TOP10平均5日收益："
        f"{average_top10_return * 100:+.2f}%"
    )

    logger.info(
        f"累计收益："
        f"{cumulative_return * 100:+.2f}%"
    )

    logger.info(
        f"年化收益："
        f"{annualized_return * 100:+.2f}%"
    )

    logger.info(
        f"最大回撤："
        f"{max_drawdown * 100:.2f}%"
    )

    logger.info(
        f"Sharpe："
        f"{sharpe:.4f}"
    )

    logger.info(
        f"盈利轮次："
        f"{profitable_rounds}"
    )

    logger.info(
        f"亏损轮次："
        f"{losing_rounds}"
    )

    logger.info(
        f"持平轮次："
        f"{flat_rounds}"
    )

    logger.info(
        f"结果文件："
        f"{RESULT_PATH}"
    )

    logger.info("=" * 70)

    # ========================================================
    # 返回结果
    # ========================================================

    return {
        "stocks": len(symbols),

        "valid_stocks":
            len(stock_data),

        "features":
            len(feature_columns),

        "backtest_days":
            len(backtest_dates),

        "rounds":
            len(result_df),

        "average_ic":
            average_ic,

        "average_rank_ic":
            average_rank_ic,

        "average_validation_ic":
            average_validation_ic,

        "average_validation_rank_ic":
            average_validation_rank_ic,

        "average_validation_rmse":
            average_validation_rmse,

        "average_validation_mae":
            average_validation_mae,

        "top10_win_rate":
            average_top10_win_rate,

        "top10_avg_return":
            average_top10_return,

        "cumulative_return":
            cumulative_return,

        "annualized_return":
            annualized_return,

        "max_drawdown":
            max_drawdown,

        "sharpe":
            sharpe,

        "profitable_rounds":
            profitable_rounds,

        "losing_rounds":
            losing_rounds,

        "flat_rounds":
            flat_rounds,

        "result_path":
            str(RESULT_PATH),
    }


# ============================================================
# 程序入口
# ============================================================


def main() -> None:
    """
    执行严格历史回测。
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

    result = run_backtest(
        engine,
        max_stocks=MAX_STOCKS,
        backtest_days=BACKTEST_DAYS,
        retrain_every=RETRAIN_EVERY,
        top_k=TOP_K,
    )

    # ========================================================
    # 控制台最终结果
    # ========================================================

    print()
    print("=" * 70)
    print(
        "Alpha158 + LightGBM "
        "严格历史回测完成"
    )
    print("=" * 70)

    print(
        f"训练股票数量："
        f"{result['stocks']}"
    )

    print(
        f"有效股票数量："
        f"{result['valid_stocks']}"
    )

    print(
        f"Alpha特征数量："
        f"{result['features']}"
    )

    print(
        f"回测交易日："
        f"{result['backtest_days']}"
    )

    print(
        f"实际回测轮次："
        f"{result['rounds']}"
    )

    print()

    print(
        f"平均IC："
        f"{result['average_ic']:.6f}"
    )

    print(
        f"平均Rank IC："
        f"{result['average_rank_ic']:.6f}"
    )

    print(
        f"验证集平均IC："
        f"{result['average_validation_ic']:.6f}"
    )

    print(
        f"验证集平均Rank IC："
        f"{result['average_validation_rank_ic']:.6f}"
    )

    print(
        f"验证集平均RMSE："
        f"{result['average_validation_rmse']:.6f}"
    )

    print(
        f"验证集平均MAE："
        f"{result['average_validation_mae']:.6f}"
    )

    print()

    print(
        f"TOP10胜率："
        f"{result['top10_win_rate'] * 100:.2f}%"
    )

    print(
        f"TOP10平均5日收益："
        f"{result['top10_avg_return'] * 100:+.2f}%"
    )

    print(
        f"累计收益："
        f"{result['cumulative_return'] * 100:+.2f}%"
    )

    print(
        f"年化收益："
        f"{result['annualized_return'] * 100:+.2f}%"
    )

    print(
        f"最大回撤："
        f"{result['max_drawdown'] * 100:.2f}%"
    )

    print(
        f"Sharpe："
        f"{result['sharpe']:.4f}"
    )

    print()

    print(
        f"盈利轮次："
        f"{result['profitable_rounds']}"
    )

    print(
        f"亏损轮次："
        f"{result['losing_rounds']}"
    )

    print(
        f"持平轮次："
        f"{result['flat_rounds']}"
    )

    print()

    print(
        f"结果文件："
        f"{result['result_path']}"
    )

    print("=" * 70)


# ============================================================
# 程序入口
# ============================================================


if __name__ == "__main__":
    main()
