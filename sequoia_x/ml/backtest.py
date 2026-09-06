"""Alpha158 + LightGBM 严格历史回测。

功能：
    1. 使用历史数据进行滚动训练
    2. 严格避免未来数据泄露
    3. 计算 IC
    4. 计算 Rank IC
    5. 计算 AI TOP10 胜率
    6. 计算 TOP10 平均未来5日收益
    7. 计算累计收益
    8. 计算最大回撤
    9. 保存逐日回测结果

说明：
    当前版本默认：
        股票数量：500
        回测周期：最近约250个交易日
        每20个交易日重新训练一次模型
        预测未来5个交易日收益

    这是一版研究/验证级回测，
    不代表实盘收益。
"""

from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from sequoia_x.core.config import get_settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.ml.alpha158 import build_alpha158_features
from sequoia_x.ml.dataset import (
    get_feature_columns,
)


logger = logging.getLogger(__name__)


# ============================================================
# 回测参数
# ============================================================

MAX_STOCKS = 500

# 最近约250个交易日作为正式回测区间
BACKTEST_DAYS = 250

# 每20个交易日重新训练一次模型
RETRAIN_EVERY = 20

# 预测未来5个交易日收益
HORIZON = 5

# 每次选TOP10
TOP_K = 10

# 最少历史数据
MIN_HISTORY = 180

# 输出文件
RESULT_PATH = Path(
    "backtest_results.csv"
)


# ============================================================
# LightGBM 参数
# ============================================================

MODEL_PARAMS = {
    "objective": "regression",
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}


# ============================================================
# 工具函数
# ============================================================


def calculate_ic(
    prediction: pd.Series,
    actual: pd.Series,
) -> float:
    """计算 Pearson IC。"""

    data = pd.DataFrame(
        {
            "prediction": prediction,
            "actual": actual,
        }
    ).dropna()

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
    """计算 Spearman Rank IC。"""

    data = pd.DataFrame(
        {
            "prediction": prediction,
            "actual": actual,
        }
    ).dropna()

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
    """计算最大回撤。"""

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


def build_stock_features(
    engine: DataEngine,
    symbols: list[str],
) -> dict[str, pd.DataFrame]:
    """一次性构建所有股票的 Alpha158 特征。"""

    logger.info("=" * 60)
    logger.info("开始构建历史 Alpha158 特征")
    logger.info("=" * 60)

    stock_data: dict[str, pd.DataFrame] = {}

    total = len(symbols)

    for index, symbol in enumerate(
        symbols,
        start=1,
    ):
        try:
            df = engine.get_ohlcv(symbol)

            if df.empty:
                continue

            if len(df) < MIN_HISTORY:
                continue

            features = build_alpha158_features(
                df
            )

            if features.empty:
                continue

            features["date"] = pd.to_datetime(
                features["date"],
                errors="coerce",
            )

            features = features.sort_values(
                "date"
            ).reset_index(
                drop=True
            )

            # 构造未来5日真实收益
            features["future_return"] = (
                features["close"].shift(-HORIZON)
                / features["close"]
                - 1.0
            )

            stock_data[symbol] = features

            if index % 50 == 0:
                logger.info(
                    f"特征构建进度："
                    f"{index}/{total}"
                )

        except Exception as exc:
            logger.warning(
                f"[{symbol}] 特征构建失败：{exc}"
            )

    logger.info(
        f"成功构建股票数量："
        f"{len(stock_data)}"
    )

    return stock_data


def get_feature_names(
    sample_df: pd.DataFrame,
) -> list[str]:
    """获取模型特征列。"""

    excluded_columns = {
        "future_return",
    }

    feature_columns = [
        column
        for column in get_feature_columns(
            sample_df
        )
        if column not in excluded_columns
    ]

    return feature_columns


def build_training_data(
    stock_data: dict[str, pd.DataFrame],
    train_end_date: pd.Timestamp,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """构造指定日期之前的训练数据。

    非常重要：

    训练样本必须满足：

        样本日期 + 5日预测周期 <= train_end_date

    从而避免标签使用未来数据。
    """

    datasets: list[pd.DataFrame] = []

    label_end_date = (
        train_end_date
        - pd.Timedelta(
            days=HORIZON * 2
        )
    )

    for symbol, df in stock_data.items():

        data = df[
            df["date"] <= label_end_date
        ].copy()

        if data.empty:
            continue

        data = data.dropna(
            subset=[
                "future_return",
            ]
        )

        if data.empty:
            continue

        required_columns = (
            feature_columns
            + ["future_return"]
        )

        data = data.dropna(
            subset=required_columns
        )

        if data.empty:
            continue

        data["symbol"] = symbol

        datasets.append(
            data[
                feature_columns
                + [
                    "future_return",
                    "symbol",
                    "date",
                ]
            ]
        )

    if not datasets:
        return (
            pd.DataFrame(),
            pd.Series(
                dtype=float
            ),
        )

    training_data = pd.concat(
        datasets,
        ignore_index=True,
    )

    X = training_data[
        feature_columns
    ].copy()

    y = pd.to_numeric(
        training_data[
            "future_return"
        ],
        errors="coerce",
    )

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

    return X, y


def train_backtest_model(
    X: pd.DataFrame,
    y: pd.Series,
) -> lgb.LGBMRegressor:
    """训练一个回测模型。"""

    if len(X) < 1000:
        raise RuntimeError(
            f"训练样本过少：{len(X)}"
        )

    split_index = int(
        len(X) * 0.8
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
    ]

    X_valid = X.iloc[
        split_index:
    ]

    y_train = y.iloc[
        :split_index
    ]

    y_valid = y.iloc[
        split_index:
    ]

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
                stopping_rounds=30,
                verbose=False,
            )
        ],
    )

    return model


def predict_one_date(
    model: lgb.LGBMRegressor,
    stock_data: dict[str, pd.DataFrame],
    prediction_date: pd.Timestamp,
    feature_columns: list[str],
) -> pd.DataFrame:
    """对某一个交易日进行横截面预测。"""

    rows: list[dict] = []

    for symbol, df in stock_data.items():

        current = df[
            df["date"]
            == prediction_date
        ]

        if current.empty:
            continue

        row = current.iloc[
            -1
        ]

        X = pd.DataFrame(
            [
                row[
                    feature_columns
                ]
            ]
        )

        X = X.apply(
            pd.to_numeric,
            errors="coerce",
        )

        if X.isna().any().any():
            continue

        try:
            prediction = float(
                model.predict(X)[0]
            )

            actual_return = float(
                row["future_return"]
            )

            if pd.isna(
                actual_return
            ):
                continue

            rows.append(
                {
                    "symbol": symbol,
                    "prediction": prediction,
                    "actual_return": actual_return,
                }
            )

        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


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
    """执行严格历史回测。"""

    logger.info("=" * 60)
    logger.info(
        "开始 Alpha158 + LightGBM 历史回测"
    )
    logger.info("=" * 60)

    symbols = engine.get_local_symbols()

    if not symbols:
        raise RuntimeError(
            "数据库中没有股票数据"
        )

    symbols = symbols[
        :max_stocks
    ]

    logger.info(
        f"回测股票数量：{len(symbols)}"
    )

    # --------------------------------------------------------
    # 1. 构建全部历史特征
    # --------------------------------------------------------

    stock_data = build_stock_features(
        engine,
        symbols,
    )

    if not stock_data:
        raise RuntimeError(
            "没有生成有效历史特征"
        )

    # --------------------------------------------------------
    # 2. 找出共同可用交易日
    # --------------------------------------------------------

    all_dates: set[pd.Timestamp] = set()

    for df in stock_data.values():

        dates = set(
            df["date"]
            .dropna()
            .tolist()
        )

        all_dates.update(
            dates
        )

    if not all_dates:
        raise RuntimeError(
            "没有找到有效交易日期"
        )

    all_dates = sorted(
        all_dates
    )

    # 需要足够历史数据之后才开始回测
    if len(all_dates) <= (
        backtest_days
        + MIN_HISTORY
    ):
        raise RuntimeError(
            "历史交易日不足，无法完成回测"
        )

    backtest_dates = all_dates[
        -backtest_days:
    ]

    # 每20个交易日选择一个回测点
    prediction_dates = (
        backtest_dates[
            ::retrain_every
        ]
    )

    logger.info(
        f"回测起始日期："
        f"{backtest_dates[0].date()}"
    )

    logger.info(
        f"回测结束日期："
        f"{backtest_dates[-1].date()}"
    )

    logger.info(
        f"回测交易日数量："
        f"{len(backtest_dates)}"
    )

    logger.info(
        f"模型重新训练次数："
        f"{len(prediction_dates)}"
    )

    # --------------------------------------------------------
    # 3. 获取特征名称
    # --------------------------------------------------------

    sample_df = next(
        iter(
            stock_data.values()
        )
    )

    feature_columns = (
        get_feature_names(
            sample_df
        )
    )

    if not feature_columns:
        raise RuntimeError(
            "没有找到模型特征"
        )

    logger.info(
        f"Alpha特征数量："
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # 4. 开始滚动回测
    # --------------------------------------------------------

    results: list[dict] = []

    for round_index, prediction_date in enumerate(
        prediction_dates,
        start=1,
    ):

        logger.info("=" * 60)
        logger.info(
            f"回测轮次："
            f"{round_index}/"
            f"{len(prediction_dates)}"
        )

        logger.info(
            f"预测日期："
            f"{prediction_date.date()}"
        )

        # ----------------------------------------------------
        # 训练截止日期
        #
        # 预测当天不能使用当天收盘之后形成的标签。
        # 因此训练标签必须早于预测日期。
        # ----------------------------------------------------

        train_end_date = (
            prediction_date
            - pd.Timedelta(
                days=HORIZON * 2
            )
        )

        logger.info(
            f"训练数据截止："
            f"{train_end_date.date()}"
        )

        X_train, y_train = (
            build_training_data(
                stock_data,
                train_end_date,
                feature_columns,
            )
        )

        if X_train.empty:
            logger.warning(
                "训练数据为空，跳过本轮"
            )
            continue

        logger.info(
            f"训练样本："
            f"{len(X_train)}"
        )

        # ----------------------------------------------------
        # 训练模型
        # ----------------------------------------------------

        model = train_backtest_model(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # 预测
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # IC
        # ----------------------------------------------------

        ic = calculate_ic(
            prediction_df[
                "prediction"
            ],
            prediction_df[
                "actual_return"
            ],
        )

        # ----------------------------------------------------
        # Rank IC
        # ----------------------------------------------------

        rank_ic = calculate_rank_ic(
            prediction_df[
                "prediction"
            ],
            prediction_df[
                "actual_return"
            ],
        )

        # ----------------------------------------------------
        # TOP10
        # ----------------------------------------------------

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

        top10_average_return = float(
            top_result[
                "actual_return"
            ].mean()
        )

        top10_win_rate = float(
            (
                top_result[
                    "actual_return"
                ]
                > 0
            ).mean()
        )

        results.append(
            {
                "date": prediction_date,
                "ic": ic,
                "rank_ic": rank_ic,
                "top10_win_rate": top10_win_rate,
                "top10_avg_return": top10_average_return,
                "top10_count": len(
                    top_result
                ),
            }
        )

        logger.info(
            f"有效股票："
            f"{len(prediction_df)}"
        )

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
            f"{top10_win_rate * 100:.2f}%"
        )

        logger.info(
            f"TOP10平均5日收益："
            f"{top10_average_return * 100:+.2f}%"
        )

    # --------------------------------------------------------
    # 5. 汇总结果
    # --------------------------------------------------------

    if not results:
        raise RuntimeError(
            "没有产生有效回测结果"
        )

    result_df = pd.DataFrame(
        results
    )

    result_df = result_df.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # 6. 累计收益
    # --------------------------------------------------------

    result_df[
        "cumulative_return"
    ] = (
        1.0
        + result_df[
            "top10_avg_return"
        ]
    ).cumprod() - 1.0

    # --------------------------------------------------------
    # 7. 最大回撤
    # --------------------------------------------------------

    result_df[
        "equity"
    ] = (
        1.0
        + result_df[
            "top10_avg_return"
        ]
    ).cumprod()

    result_df[
        "running_max"
    ] = result_df[
        "equity"
    ].cummax()

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

    # --------------------------------------------------------
    # 8. 汇总指标
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 9. 保存结果
    # --------------------------------------------------------

    result_df[
        "date"
    ] = result_df[
        "date"
    ].dt.strftime(
        "%Y-%m-%d"
    )

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        RESULT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 10. 输出最终结果
    # --------------------------------------------------------

    logger.info("")
    logger.info("=" * 60)
    logger.info(
        "Alpha158 + LightGBM 历史回测完成"
    )
    logger.info("=" * 60)

    logger.info(
        f"训练股票：{len(symbols)}"
    )

    logger.info(
        f"有效股票：{len(stock_data)}"
    )

    logger.info(
        f"回测轮次：{len(result_df)}"
    )

    logger.info(
        f"平均 IC："
        f"{average_ic:.6f}"
    )

    logger.info(
        f"平均 Rank IC："
        f"{average_rank_ic:.6f}"
    )

    logger.info(
        f"TOP10 胜率："
        f"{average_top10_win_rate * 100:.2f}%"
    )

    logger.info(
        f"TOP10 平均5日收益："
        f"{average_top10_return * 100:+.2f}%"
    )

    logger.info(
        f"累计收益："
        f"{cumulative_return * 100:+.2f}%"
    )

    logger.info(
        f"最大回撤："
        f"{max_drawdown * 100:.2f}%"
    )

    logger.info(
        f"详细结果："
        f"{RESULT_PATH}"
    )

    logger.info("=" * 60)

    return {
        "stocks": len(symbols),
        "valid_stocks": len(
            stock_data
        ),
        "rounds": len(
            result_df
        ),
        "average_ic": average_ic,
        "average_rank_ic": average_rank_ic,
        "top10_win_rate": (
            average_top10_win_rate
        ),
        "top10_avg_return": (
            average_top10_return
        ),
        "cumulative_return": (
            cumulative_return
        ),
        "max_drawdown": (
            max_drawdown
        ),
        "result_path": str(
            RESULT_PATH
        ),
    }


# ============================================================
# 程序入口
# ============================================================


def main() -> None:

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

    result = run_backtest(
        engine,
        max_stocks=MAX_STOCKS,
        backtest_days=BACKTEST_DAYS,
        retrain_every=RETRAIN_EVERY,
        top_k=TOP_K,
    )

    print()
    print("=" * 60)
    print(
        "Alpha158 + LightGBM 历史回测完成"
    )
    print("=" * 60)

    print(
        f"训练股票："
        f"{result['stocks']}"
    )

    print(
        f"有效股票："
        f"{result['valid_stocks']}"
    )

    print(
        f"回测轮次："
        f"{result['rounds']}"
    )

    print(
        f"平均 IC："
        f"{result['average_ic']:.6f}"
    )

    print(
        f"平均 Rank IC："
        f"{result['average_rank_ic']:.6f}"
    )

    print(
        f"TOP10 胜率："
        f"{result['top10_win_rate'] * 100:.2f}%"
    )

    print(
        f"TOP10 平均5日收益："
        f"{result['top10_avg_return'] * 100:+.2f}%"
    )

    print(
        f"累计收益："
        f"{result['cumulative_return'] * 100:+.2f}%"
    )

    print(
        f"最大回撤："
        f"{result['max_drawdown'] * 100:.2f}%"
    )

    print(
        f"结果文件："
        f"{result['result_path']}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
