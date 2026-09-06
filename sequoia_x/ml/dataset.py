"""LightGBM 训练数据集构建。

功能：
    1. 读取单只股票历史 OHLCV
    2. 调用 Alpha158 风格特征工程
    3. 构造未来 5 个交易日收益率
    4. 生成 LightGBM 可使用的训练数据

注意：
    当前阶段只负责数据集构建，不训练模型。
"""

from __future__ import annotations

import pandas as pd

from sequoia_x.ml.alpha158 import build_alpha158_features


def build_training_dataset(
    df: pd.DataFrame,
    horizon: int = 5,
) -> pd.DataFrame:
    """构建单只股票的 LightGBM 训练数据。

    参数：
        df:
            单只股票历史行情。
        horizon:
            预测未来多少个交易日，默认 5 天。

    返回：
        包含 Alpha 特征和未来收益标签的数据。
    """

    if df.empty:
        return pd.DataFrame()

    if horizon <= 0:
        raise ValueError("horizon 必须大于 0")

    data = build_alpha158_features(df)

    if data.empty:
        return pd.DataFrame()

    # =========================================================
    # 构造预测目标
    # =========================================================
    #
    # 今天收盘价 -> 未来第 5 个交易日收盘价
    #
    # 例如：
    #
    # 今天：100 元
    # 5日后：105 元
    #
    # target = 105 / 100 - 1 = 5%
    #
    data["target_return_5d"] = (
        data["close"].shift(-horizon)
        / data["close"]
        - 1
    )

    # =========================================================
    # 删除最后 horizon 天
    # =========================================================
    #
    # 因为最后几天没有未来数据，
    # 所以无法计算 target。
    #

    data = data.dropna(
        subset=["target_return_5d"]
    )

    # =========================================================
    # 删除没有足够历史数据的早期记录
    # =========================================================
    #
    # Alpha158 风格特征中最长使用 120 日窗口。
    #
    # 前 120 个交易日特征不完整，
    # 因此不拿来训练。
    #

    if len(data) <= 120:
        return pd.DataFrame()

    data = data.iloc[120:].copy()

    # =========================================================
    # 清理无限值
    # =========================================================

    data = data.replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )

    # =========================================================
    # 删除 target 或关键特征为空的记录
    # =========================================================

    data = data.dropna(
        subset=["target_return_5d"]
    )

    return data.reset_index(drop=True)


def get_feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    """获取可以交给 LightGBM 的特征列。

    会自动排除：
        原始行情字段
        日期
        股票代码
        target
    """

    excluded_columns = {
        "id",
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
        "target_return_5d",
    }

    return [
        column
        for column in df.columns
        if column not in excluded_columns
    ]


def prepare_lightgbm_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """准备 LightGBM 的 X、y 和特征名称。

    返回：
        X:
            特征矩阵
        y:
            未来 5 日收益
        feature_columns:
            特征名称列表
    """

    if df.empty:
        return (
            pd.DataFrame(),
            pd.Series(dtype=float),
            [],
        )

    feature_columns = get_feature_columns(df)

    if not feature_columns:
        return (
            pd.DataFrame(),
            pd.Series(dtype=float),
            [],
        )

    X = df[feature_columns].copy()

    y = pd.to_numeric(
        df["target_return_5d"],
        errors="coerce",
    )

    # LightGBM 不能直接处理 NaN/无限值之外的异常数据，
    # 这里统一清理。
    X = X.replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )

    valid_mask = y.notna()

    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].copy()

    return (
        X,
        y,
        feature_columns,
    )
