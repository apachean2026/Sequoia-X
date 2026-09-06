"""Alpha158 + LightGBM 模型训练器。

功能：
    1. 从 Sequoia-X 本地数据库读取历史 OHLCV
    2. 构建 Alpha158 风格特征
    3. 构造未来 5 个交易日收益标签
    4. 按时间严格切分训练集 / 验证集
    5. 使用 LightGBM 训练回归模型
    6. 保存模型和特征名称
    7. 输出基础验证指标

重要原则：
    - 不随机打乱时间序列
    - 不使用未来数据构造特征
    - 训练集始终早于验证集
    - 模型训练阶段不参与严格回测
    - 严格回测将在后续模块单独完成
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from sequoia_x.core.config import get_settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.ml.dataset import (
    build_training_dataset,
    prepare_lightgbm_data,
)


logger = logging.getLogger(__name__)


# =========================================================
# 模型保存路径
# =========================================================

MODEL_DIR = Path("models")

MODEL_PATH = (
    MODEL_DIR
    / "alpha158_lightgbm.txt"
)

FEATURE_PATH = (
    MODEL_DIR
    / "alpha158_features.json"
)


# =========================================================
# 默认参数
# =========================================================

DEFAULT_HORIZON = 5

DEFAULT_MAX_STOCKS = 500

TRAIN_RATIO = 0.8


# =========================================================
# 工具函数
# =========================================================

def _calculate_metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
) -> dict[str, float]:
    """计算验证集基础指标。"""

    y_true = pd.to_numeric(
        y_true,
        errors="coerce",
    )

    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    valid_mask = (
        y_true.notna()
        & np.isfinite(predictions)
    )

    if valid_mask.sum() == 0:
        return {
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "directional_accuracy": float("nan"),
            "correlation": float("nan"),
        }

    actual = y_true.loc[
        valid_mask
    ].to_numpy(
        dtype=float
    )

    pred = predictions[
        valid_mask.to_numpy()
    ]

    errors = pred - actual

    mse = float(
        np.mean(
            errors ** 2
        )
    )

    rmse = float(
        np.sqrt(mse)
    )

    mae = float(
        np.mean(
            np.abs(errors)
        )
    )

    # -----------------------------------------------------
    # 方向准确率
    # -----------------------------------------------------

    actual_direction = (
        actual > 0
    )

    predicted_direction = (
        pred > 0
    )

    directional_accuracy = float(
        np.mean(
            actual_direction
            == predicted_direction
        )
    )

    # -----------------------------------------------------
    # Pearson 相关系数
    # -----------------------------------------------------

    if (
        len(actual) >= 2
        and np.std(actual) > 0
        and np.std(pred) > 0
    ):
        correlation = float(
            np.corrcoef(
                actual,
                pred,
            )[0, 1]
        )
    else:
        correlation = float("nan")

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "directional_accuracy": (
            directional_accuracy
        ),
        "correlation": correlation,
    }


def _save_feature_names(
    feature_columns: list[str],
) -> None:
    """保存训练时使用的特征名称。

    后续预测阶段必须使用完全一致的特征顺序。
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "feature_count": len(
            feature_columns
        ),
        "features": feature_columns,
    }

    with FEATURE_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(
        f"特征列表已保存：{FEATURE_PATH}"
    )


def _prepare_training_data(
    datasets: list[pd.DataFrame],
) -> pd.DataFrame:
    """合并并严格按照日期排序训练数据。"""

    if not datasets:
        return pd.DataFrame()

    data = pd.concat(
        datasets,
        ignore_index=True,
    )

    if data.empty:
        return pd.DataFrame()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce",
    )

    data = data.dropna(
        subset=["date"]
    ).copy()

    # -----------------------------------------------------
    # 时间排序
    # -----------------------------------------------------

    data = (
        data
        .sort_values(
            ["date", "symbol"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    return data


# =========================================================
# 主训练函数
# =========================================================

def train_model(
    engine: DataEngine,
    max_stocks: int = DEFAULT_MAX_STOCKS,
    train_ratio: float = TRAIN_RATIO,
    horizon: int = DEFAULT_HORIZON,
) -> dict:
    """训练 Alpha158 + LightGBM 模型。

    参数：
        engine:
            Sequoia-X 数据引擎。

        max_stocks:
            最多使用多少只股票。
            第一阶段建议 500。

        train_ratio:
            时间序列训练集比例。
            默认 80%。

        horizon:
            预测未来多少个交易日。
            默认 5 个交易日。

    返回：
        包含训练结果和验证结果的字典。
    """

    logger.info("=" * 70)
    logger.info(
        "开始训练 Alpha158 + LightGBM 模型"
    )
    logger.info("=" * 70)

    # =====================================================
    # 参数检查
    # =====================================================

    if max_stocks <= 0:
        raise ValueError(
            "max_stocks 必须大于 0"
        )

    if not 0.5 <= train_ratio < 1:
        raise ValueError(
            "train_ratio 必须在 0.5 到 1 之间"
        )

    if horizon <= 0:
        raise ValueError(
            "horizon 必须大于 0"
        )

    # =====================================================
    # 获取本地股票
    # =====================================================

    symbols = engine.get_local_symbols()

    if not symbols:
        raise RuntimeError(
            "数据库中没有股票数据"
        )

    symbols = list(
        symbols[:max_stocks]
    )

    logger.info(
        f"本次训练股票数量："
        f"{len(symbols)}"
    )

    # =====================================================
    # 构建训练数据
    # =====================================================

    datasets: list[pd.DataFrame] = []

    successful_stocks = 0

    for index, symbol in enumerate(
        symbols,
        start=1,
    ):
        try:
            logger.info(
                f"[{index}/{len(symbols)}] "
                f"处理股票：{symbol}"
            )

            df = engine.get_ohlcv(
                symbol
            )

            if df.empty:
                logger.warning(
                    f"[{symbol}] "
                    f"没有行情数据"
                )
                continue

            dataset = build_training_dataset(
                df,
                horizon=horizon,
            )

            if dataset.empty:
                logger.warning(
                    f"[{symbol}] "
                    f"训练数据不足"
                )
                continue

            # 股票代码必须保留，
            # 后续用于时间排序和分析。
            dataset["symbol"] = symbol

            datasets.append(
                dataset
            )

            successful_stocks += 1

            logger.info(
                f"[{index}/{len(symbols)}] "
                f"{symbol}："
                f"{len(dataset)} 条训练数据"
            )

        except Exception as exc:
            logger.exception(
                f"[{symbol}] "
                f"数据处理失败：{exc}"
            )

    # =====================================================
    # 检查训练数据
    # =====================================================

    if not datasets:
        raise RuntimeError(
            "没有生成任何有效训练数据"
        )

    data = _prepare_training_data(
        datasets
    )

    if data.empty:
        raise RuntimeError(
            "合并后的训练数据为空"
        )

    logger.info("=" * 70)
    logger.info(
        f"成功生成训练数据的股票："
        f"{successful_stocks}"
    )
    logger.info(
        f"训练数据总量：{len(data)}"
    )
    logger.info(
        f"数据开始日期："
        f"{data['date'].min()}"
    )
    logger.info(
        f"数据结束日期："
        f"{data['date'].max()}"
    )
    logger.info("=" * 70)

    # =====================================================
    # 准备 LightGBM X / y
    # =====================================================

    X, y, feature_columns = (
        prepare_lightgbm_data(
            data
        )
    )

    if X.empty or y.empty:
        raise RuntimeError(
            "LightGBM 训练数据为空"
        )

    if not feature_columns:
        raise RuntimeError(
            "没有可用的 Alpha 特征"
        )

    logger.info(
        f"Alpha特征数量："
        f"{len(feature_columns)}"
    )

    logger.info(
        f"有效样本数量："
        f"{len(X)}"
    )

    # =====================================================
    # 重新对齐日期
    # =====================================================
    #
    # 注意：
    # prepare_lightgbm_data() 会清理无效 target，
    # 并重新生成 X 的索引。
    #
    # 因此不能再使用：
    #
    # data.loc[X.index]
    #
    # 去猜测日期。
    #
    # 这里直接根据原始 data 构造一个完全对应的
    # target / feature 数据集，然后统一排序。
    # =====================================================

    modeling_data = data.copy()

    modeling_data["date"] = pd.to_datetime(
        modeling_data["date"],
        errors="coerce",
    )

    modeling_data = modeling_data.dropna(
        subset=["date"]
    ).copy()

    # -----------------------------------------------------
    # 只保留真正需要的列
    # -----------------------------------------------------

    available_features = [
        column
        for column in feature_columns
        if column in modeling_data.columns
    ]

    if not available_features:
        raise RuntimeError(
            "训练特征与原始数据无法对应"
        )

    modeling_data = modeling_data[
        [
            "date",
            "symbol",
            *available_features,
            "target_return_5d",
        ]
    ].copy()

    # -----------------------------------------------------
    # 数值化
    # -----------------------------------------------------

    for column in available_features:
        modeling_data[column] = pd.to_numeric(
            modeling_data[column],
            errors="coerce",
        )

    modeling_data[
        "target_return_5d"
    ] = pd.to_numeric(
        modeling_data[
            "target_return_5d"
        ],
        errors="coerce",
    )

    # -----------------------------------------------------
    # 清理 target
    # -----------------------------------------------------

    modeling_data = (
        modeling_data
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=[
                "target_return_5d",
            ]
        )
        .copy()
    )

    # =====================================================
    # 严格时间排序
    # =====================================================

    modeling_data = (
        modeling_data
        .sort_values(
            ["date", "symbol"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    if len(modeling_data) < 100:
        raise RuntimeError(
            "有效训练样本太少，"
            "无法进行可靠的训练/验证切分"
        )

    # =====================================================
    # 构造 X / y
    # =====================================================

    X_model = modeling_data[
        available_features
    ].copy()

    y_model = modeling_data[
        "target_return_5d"
    ].copy()

    # =====================================================
    # 再次清理异常值
    # =====================================================

    X_model = X_model.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    y_model = y_model.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # -----------------------------------------------------
    # 这里允许特征存在 NaN。
    #
    # LightGBM 本身可以处理缺失值。
    # 我们不把 NaN 行全部删除，
    # 避免不必要地损失训练样本。
    # -----------------------------------------------------

    valid_target_mask = (
        y_model.notna()
    )

    X_model = X_model.loc[
        valid_target_mask
    ].reset_index(drop=True)

    y_model = y_model.loc[
        valid_target_mask
    ].reset_index(drop=True)

    modeling_data = modeling_data.loc[
        valid_target_mask
    ].reset_index(drop=True)

    # =====================================================
    # 检查最终特征数量
    # =====================================================

    if X_model.empty:
        raise RuntimeError(
            "最终 X 数据为空"
        )

    if y_model.empty:
        raise RuntimeError(
            "最终 y 数据为空"
        )

    logger.info(
        f"最终训练样本："
        f"{len(X_model)}"
    )

    logger.info(
        f"最终特征数量："
        f"{len(available_features)}"
    )

    # =====================================================
    # 时间序列切分
    # =====================================================
    #
    # 前 80%：
    #     训练集
    #
    # 后 20%：
    #     验证集
    #
    # 不随机打乱。
    # =====================================================

    split_index = int(
        len(X_model)
        * train_ratio
    )

    if (
        split_index <= 0
        or split_index >= len(X_model)
    ):
        raise RuntimeError(
            "训练集/验证集切分失败"
        )

    X_train = X_model.iloc[
        :split_index
    ].copy()

    X_valid = X_model.iloc[
        split_index:
    ].copy()

    y_train = y_model.iloc[
        :split_index
    ].copy()

    y_valid = y_model.iloc[
        split_index:
    ].copy()

    train_data = modeling_data.iloc[
        :split_index
    ].copy()

    valid_data = modeling_data.iloc[
        split_index:
    ].copy()

    logger.info("=" * 70)
    logger.info(
        "时间序列切分完成"
    )
    logger.info(
        f"训练集：{len(X_train)}"
    )
    logger.info(
        f"验证集：{len(X_valid)}"
    )
    logger.info(
        f"训练开始："
        f"{train_data['date'].min()}"
    )
    logger.info(
        f"训练结束："
        f"{train_data['date'].max()}"
    )
    logger.info(
        f"验证开始："
        f"{valid_data['date'].min()}"
    )
    logger.info(
        f"验证结束："
        f"{valid_data['date'].max()}"
    )
    logger.info("=" * 70)

    # =====================================================
    # LightGBM 模型
    # =====================================================

    model = lgb.LGBMRegressor(
        objective="regression",

        # -------------------------------------------------
        # 基础模型参数
        # -------------------------------------------------

        n_estimators=1000,

        learning_rate=0.03,

        num_leaves=31,

        max_depth=-1,

        min_child_samples=30,

        subsample=0.8,

        subsample_freq=1,

        colsample_bytree=0.8,

        reg_alpha=0.1,

        reg_lambda=0.1,

        # -------------------------------------------------
        # 随机种子
        # -------------------------------------------------

        random_state=42,

        # -------------------------------------------------
        # CPU
        # -------------------------------------------------

        n_jobs=-1,

        verbosity=-1,
    )

    logger.info("=" * 70)
    logger.info(
        "开始 LightGBM 训练..."
    )
    logger.info("=" * 70)

    # =====================================================
    # 模型训练
    # =====================================================

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
                period=50
            ),
        ],
    )

    # =====================================================
    # 验证集预测
    # =====================================================

    predictions = model.predict(
        X_valid,
        num_iteration=(
            model.best_iteration_
            if model.best_iteration_
            else None
        ),
    )

    # =====================================================
    # 验证指标
    # =====================================================

    metrics = _calculate_metrics(
        y_valid,
        predictions,
    )

    mse = metrics["mse"]

    rmse = metrics["rmse"]

    mae = metrics["mae"]

    directional_accuracy = (
        metrics[
            "directional_accuracy"
        ]
    )

    correlation = metrics[
        "correlation"
    ]

    logger.info("=" * 70)
    logger.info(
        "LightGBM 训练完成"
    )
    logger.info(
        f"MSE：{mse:.8f}"
    )
    logger.info(
        f"RMSE：{rmse:.6f}"
    )
    logger.info(
        f"MAE：{mae:.6f}"
    )
    logger.info(
        f"方向准确率："
        f"{directional_accuracy:.4%}"
    )

    if np.isfinite(correlation):
        logger.info(
            f"预测相关系数："
            f"{correlation:.6f}"
        )
    else:
        logger.info(
            "预测相关系数：N/A"
        )

    logger.info(
        f"最佳迭代次数："
        f"{model.best_iteration_}"
    )

    logger.info("=" * 70)

    # =====================================================
    # 特征重要性
    # =====================================================

    try:
        importance = pd.DataFrame(
            {
                "feature": available_features,
                "importance": (
                    model.booster_
                    .feature_importance(
                        importance_type="gain"
                    )
                ),
            }
        )

        importance = (
            importance
            .sort_values(
                "importance",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        logger.info(
            "Top 20 特征："
        )

        for _, row in importance.head(
            20
        ).iterrows():
            logger.info(
                f"  {row['feature']}: "
                f"{row['importance']:.4f}"
            )

    except Exception as exc:
        logger.warning(
            f"特征重要性计算失败：{exc}"
        )

    # =====================================================
    # 保存模型
    # =====================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.booster_.save_model(
        str(MODEL_PATH)
    )

    logger.info(
        f"模型已保存："
        f"{MODEL_PATH}"
    )

    # =====================================================
    # 保存特征列表
    # =====================================================

    _save_feature_names(
        available_features
    )

    # =====================================================
    # 返回训练结果
    # =====================================================

    result = {
        "stocks_requested": len(symbols),

        "stocks": successful_stocks,

        "samples": len(X_model),

        "features": len(
            available_features
        ),

        "train_samples": len(
            X_train
        ),

        "valid_samples": len(
            X_valid
        ),

        "train_start": str(
            train_data["date"].min()
        ),

        "train_end": str(
            train_data["date"].max()
        ),

        "valid_start": str(
            valid_data["date"].min()
        ),

        "valid_end": str(
            valid_data["date"].max()
        ),

        "mse": mse,

        "rmse": rmse,

        "mae": mae,

        "directional_accuracy": (
            directional_accuracy
        ),

        "correlation": correlation,

        "best_iteration": (
            model.best_iteration_
        ),

        "model_path": str(
            MODEL_PATH
        ),

        "feature_path": str(
            FEATURE_PATH
        ),
    }

    return result


# =========================================================
# 命令行入口
# =========================================================

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

    logger.info(
        "=" * 70
    )

    logger.info(
        "Alpha158 + LightGBM"
    )

    logger.info(
        "模型训练程序启动"
    )

    logger.info(
        "=" * 70
    )

    # -----------------------------------------------------
    # 加载 Sequoia-X 配置
    # -----------------------------------------------------

    settings = get_settings()

    # -----------------------------------------------------
    # 初始化数据引擎
    # -----------------------------------------------------

    engine = DataEngine(
        settings
    )

    # -----------------------------------------------------
    # 开始训练
    # -----------------------------------------------------

    result = train_model(
        engine,

        # 第一阶段：
        # 先使用 500 只股票。
        max_stocks=500,

        # 未来 5 个交易日收益。
        horizon=5,

        # 80% 时间训练，
        # 20% 时间验证。
        train_ratio=0.8,
    )

    # -----------------------------------------------------
    # 输出结果
    # -----------------------------------------------------

    print()

    print("=" * 70)

    print(
        "Alpha158 + LightGBM "
        "训练成功"
    )

    print("=" * 70)

    print(
        f"请求股票："
        f"{result['stocks_requested']}"
    )

    print(
        f"成功股票："
        f"{result['stocks']}"
    )

    print(
        f"训练样本："
        f"{result['samples']}"
    )

    print(
        f"Alpha特征："
        f"{result['features']}"
    )

    print(
        f"训练集："
        f"{result['train_samples']}"
    )

    print(
        f"验证集："
        f"{result['valid_samples']}"
    )

    print(
        f"训练开始："
        f"{result['train_start']}"
    )

    print(
        f"训练结束："
        f"{result['train_end']}"
    )

    print(
        f"验证开始："
        f"{result['valid_start']}"
    )

    print(
        f"验证结束："
        f"{result['valid_end']}"
    )

    print(
        f"RMSE："
        f"{result['rmse']:.6f}"
    )

    print(
        f"MAE："
        f"{result['mae']:.6f}"
    )

    print(
        f"方向准确率："
        f"{result['directional_accuracy']:.4%}"
    )

    if np.isfinite(
        result["correlation"]
    ):
        print(
            f"预测相关系数："
            f"{result['correlation']:.6f}"
        )
    else:
        print(
            "预测相关系数：N/A"
        )

    print(
        f"最佳迭代："
        f"{result['best_iteration']}"
    )

    print(
        f"模型："
        f"{result['model_path']}"
    )

    print(
        f"特征列表："
        f"{result['feature_path']}"
    )

    print("=" * 70)
