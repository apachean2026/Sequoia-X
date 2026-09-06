"""LightGBM 训练数据集构建。

功能：
    1. 读取单只股票历史 OHLCV
    2. 调用 Alpha158 风格特征工程
    3. 构造未来 N 个交易日收益率
    4. 清理无效特征
    5. 生成 LightGBM 可使用的训练数据

注意：
    当前模块只负责数据集构建，不训练模型。

数据泄漏控制：
    target 使用未来数据，但 target 只作为训练标签。
    所有特征均由当前及历史行情计算，不使用未来数据。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sequoia_x.ml.alpha158 import build_alpha158_features


# =========================================================
# 常量
# =========================================================

DEFAULT_HORIZON = 5

# Alpha158 风格特征目前最长使用 120 个交易日窗口。
# 为保证训练样本具有完整历史信息，至少保留这些历史数据。
MIN_HISTORY = 120

# 原始字段。
RAW_COLUMNS = {
    "id",
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover",
}

# 训练标签。
TARGET_COLUMN = "target_return_5d"


# =========================================================
# 数据清理工具
# =========================================================

def _clean_numeric_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """将 DataFrame 中的数据统一转换为数值并清理异常值。"""

    if df.empty:
        return df.copy()

    data = df.copy()

    # 所有非日期字段尝试转换成数值。
    for column in data.columns:
        if column == "date":
            continue

        if column in {"symbol"}:
            continue

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    # 无限值统一为 NaN。
    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return data


def _remove_invalid_feature_columns(
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """删除完全为空或没有有效变化的特征。

    LightGBM 不需要这些没有有效信息的列。
    """

    if X.empty:
        return (
            pd.DataFrame(),
            [],
        )

    data = X.copy()

    # 只保留数值列。
    numeric_columns = data.select_dtypes(
        include=[np.number],
    ).columns.tolist()

    data = data[numeric_columns].copy()

    if data.empty:
        return (
            pd.DataFrame(),
            [],
        )

    # 删除完全为空的列。
    valid_columns = [
        column
        for column in data.columns
        if data[column].notna().any()
    ]

    data = data[valid_columns].copy()

    if data.empty:
        return (
            pd.DataFrame(),
            [],
        )

    # 删除完全没有变化的常数列。
    variable_columns = [
        column
        for column in data.columns
        if data[column].nunique(
            dropna=True,
        ) > 1
    ]

    data = data[variable_columns].copy()

    return (
        data,
        variable_columns,
    )


# =========================================================
# 构造训练数据集
# =========================================================

def build_training_dataset(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    """构建单只股票的 LightGBM 训练数据。

    参数：
        df:
            单只股票历史行情。

        horizon:
            预测未来多少个交易日。
            默认 5 个交易日。

    返回：
        包含 Alpha 特征和未来收益标签的数据。

    例如：

        今天 close = 100
        未来第 5 个交易日 close = 105

        target_return_5d = 105 / 100 - 1
                         = 0.05
    """

    if df.empty:
        return pd.DataFrame()

    if horizon <= 0:
        raise ValueError(
            "horizon 必须大于 0"
        )

    # =====================================================
    # 1. 基础数据复制
    # =====================================================

    source = df.copy()

    required_columns = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = (
        required_columns
        - set(source.columns)
    )

    if missing:
        raise ValueError(
            f"缺少必要字段：{sorted(missing)}"
        )

    # =====================================================
    # 2. 日期和价格数据标准化
    # =====================================================

    source["date"] = pd.to_datetime(
        source["date"],
        errors="coerce",
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    if "turnover" in source.columns:
        numeric_columns.append(
            "turnover"
        )

    for column in numeric_columns:
        source[column] = pd.to_numeric(
            source[column],
            errors="coerce",
        )

    source = (
        source
        .dropna(
            subset=[
                "date",
                "close",
            ]
        )
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if source.empty:
        return pd.DataFrame()

    # =====================================================
    # 3. Alpha158 风格特征
    # =====================================================

    data = build_alpha158_features(
        source
    )

    if data.empty:
        return pd.DataFrame()

    # =====================================================
    # 4. 构造未来收益标签
    # =====================================================

    # 今天收盘价
    current_close = pd.to_numeric(
        data["close"],
        errors="coerce",
    )

    # 未来 horizon 个交易日收盘价
    future_close = pd.to_numeric(
        data["close"].shift(-horizon),
        errors="coerce",
    )

    data["target_return_5d"] = (
        future_close
        / current_close
        - 1
    )

    # =====================================================
    # 5. 删除没有未来标签的记录
    # =====================================================

    data = data.dropna(
        subset=[
            "target_return_5d",
        ]
    ).copy()

    if data.empty:
        return pd.DataFrame()

    # =====================================================
    # 6. 删除历史不足的数据
    # =====================================================

    if len(data) <= MIN_HISTORY:
        return pd.DataFrame()

    data = (
        data
        .iloc[MIN_HISTORY:]
        .copy()
    )

    # =====================================================
    # 7. 清理异常值
    # =====================================================

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # =====================================================
    # 8. 确保 target 为有效数值
    # =====================================================

    data["target_return_5d"] = pd.to_numeric(
        data["target_return_5d"],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            "target_return_5d",
        ]
    ).copy()

    if data.empty:
        return pd.DataFrame()

    # =====================================================
    # 9. 保持时间顺序
    # =====================================================

    data = (
        data
        .sort_values("date")
        .reset_index(drop=True)
    )

    return data


# =========================================================
# 获取 LightGBM 特征列
# =========================================================

def get_feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    """获取可以交给 LightGBM 的特征列。

    自动排除：

        id
        symbol
        date
        原始 OHLCV
        turnover
        target

    同时只返回数值型特征。
    """

    if df.empty:
        return []

    excluded_columns = (
        RAW_COLUMNS
        | {
            TARGET_COLUMN,
        }
    )

    feature_columns: list[str] = []

    for column in df.columns:

        if column in excluded_columns:
            continue

        # 只允许数值型特征进入 LightGBM。
        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):
            continue

        # 排除完全为空的特征。
        if not df[column].notna().any():
            continue

        # 排除没有变化的常数特征。
        if (
            df[column]
            .nunique(dropna=True)
            <= 1
        ):
            continue

        feature_columns.append(column)

    return feature_columns


# =========================================================
# 准备 LightGBM 数据
# =========================================================

def prepare_lightgbm_data(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    list[str],
]:
    """准备 LightGBM 的 X、y 和特征名称。

    返回：

        X:
            LightGBM 特征矩阵。

        y:
            未来 5 日收益率。

        feature_columns:
            特征名称列表。
    """

    if df.empty:
        return (
            pd.DataFrame(),
            pd.Series(
                dtype=float
            ),
            [],
        )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"缺少训练目标字段："
            f"{TARGET_COLUMN}"
        )

    # =====================================================
    # 1. 获取候选特征
    # =====================================================

    feature_columns = get_feature_columns(
        df
    )

    if not feature_columns:
        return (
            pd.DataFrame(),
            pd.Series(
                dtype=float
            ),
            [],
        )

    # =====================================================
    # 2. 构造 X
    # =====================================================

    X = df[
        feature_columns
    ].copy()

    # 强制转换为数值。
    for column in X.columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    # 清理无限值。
    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # =====================================================
    # 3. 构造 y
    # =====================================================

    y = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce",
    )

    # =====================================================
    # 4. 删除 target 无效的数据
    # =====================================================

    valid_target_mask = y.notna()

    X = X.loc[
        valid_target_mask
    ].copy()

    y = y.loc[
        valid_target_mask
    ].copy()

    if X.empty:
        return (
            pd.DataFrame(),
            pd.Series(
                dtype=float
            ),
            [],
        )

    # =====================================================
    # 5. 删除完全为空的特征
    # =====================================================

    valid_columns = [
        column
        for column in X.columns
        if X[column].notna().any()
    ]

    X = X[
        valid_columns
    ].copy()

    feature_columns = valid_columns

    if X.empty:
        return (
            pd.DataFrame(),
            pd.Series(
                dtype=float
            ),
            [],
        )

    # =====================================================
    # 6. 删除常数特征
    # =====================================================

    variable_columns = [
        column
        for column in X.columns
        if X[column].nunique(
            dropna=True
        ) > 1
    ]

    X = X[
        variable_columns
    ].copy()

    feature_columns = variable_columns

    if X.empty:
        return (
            pd.DataFrame(),
            pd.Series(
                dtype=float
            ),
            [],
        )

    # =====================================================
    # 7. 最终清理
    # =====================================================

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    y = y.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # target 必须有效。
    final_mask = y.notna()

    X = X.loc[
        final_mask
    ].copy()

    y = y.loc[
        final_mask
    ].copy()

    # =====================================================
    # 8. 重置索引
    # =====================================================

    X = X.reset_index(
        drop=True
    )

    y = y.reset_index(
        drop=True
    )

    return (
        X,
        y,
        feature_columns,
    )


# =========================================================
# 多股票训练数据辅助函数
# =========================================================

def combine_training_datasets(
    datasets: list[pd.DataFrame],
) -> pd.DataFrame:
    """合并多个股票的训练数据。

    每只股票应先独立调用：

        build_training_dataset()

    再使用本函数合并。

    这样可以避免不同股票之间计算滚动指标时发生串行污染。
    """

    valid_datasets = [
        data
        for data in datasets
        if data is not None
        and not data.empty
    ]

    if not valid_datasets:
        return pd.DataFrame()

    combined = pd.concat(
        valid_datasets,
        axis=0,
        ignore_index=True,
    )

    if "date" in combined.columns:
        combined["date"] = pd.to_datetime(
            combined["date"],
            errors="coerce",
        )

        combined = (
            combined
            .sort_values("date")
            .reset_index(drop=True)
        )

    return combined


# =========================================================
# 数据质量检查
# =========================================================

def validate_training_dataset(
    df: pd.DataFrame,
) -> dict[str, object]:
    """检查训练数据质量。

    返回简单的质量统计信息，
    方便后续训练和严格回测阶段使用。
    """

    if df.empty:
        return {
            "rows": 0,
            "features": 0,
            "target_valid_rows": 0,
            "target_mean": None,
            "target_std": None,
            "date_min": None,
            "date_max": None,
        }

    feature_columns = get_feature_columns(
        df
    )

    target = pd.to_numeric(
        df.get(
            TARGET_COLUMN,
            pd.Series(dtype=float),
        ),
        errors="coerce",
    )

    dates = pd.to_datetime(
        df.get(
            "date",
            pd.Series(dtype="datetime64[ns]"),
        ),
        errors="coerce",
    )

    return {
        "rows": int(len(df)),
        "features": int(
            len(feature_columns)
        ),
        "target_valid_rows": int(
            target.notna().sum()
        ),
        "target_mean": (
            float(target.mean())
            if target.notna().any()
            else None
        ),
        "target_std": (
            float(target.std())
            if target.notna().any()
            else None
        ),
        "date_min": (
            dates.min()
            if dates.notna().any()
            else None
        ),
        "date_max": (
            dates.max()
            if dates.notna().any()
            else None
        ),
    }
