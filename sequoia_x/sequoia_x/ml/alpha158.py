"""Alpha158 风格特征工程。

第一阶段目标：
    从 Sequoia-X 现有 stock_daily 数据中，
    计算适合 LightGBM 使用的量价特征。

说明：
    这里不依赖 Qlib，直接使用 Sequoia-X 已有的 OHLCV 数据。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """安全除法，避免除以 0。"""
    return a / b.replace(0, np.nan)


def build_alpha158_features(df: pd.DataFrame) -> pd.DataFrame:
    """根据单只股票历史 OHLCV 数据生成 Alpha158 风格特征。

    输入：
        date
        open
        high
        low
        close
        volume
        turnover（可选）

    输出：
        原始行情 + 特征列
    """

    if df.empty:
        return pd.DataFrame()

    data = df.copy()

    required_columns = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            f"缺少必要字段：{sorted(missing)}"
        )

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce",
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    if "turnover" in data.columns:
        numeric_columns.append("turnover")

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.sort_values("date").reset_index(drop=True)

    close = data["close"]
    open_price = data["open"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"]

    # =========================================================
    # 1. 收益率
    # =========================================================

    for period in [1, 3, 5, 10, 20, 60, 120]:
        data[f"RET_{period}"] = (
            close / close.shift(period) - 1
        )

    # =========================================================
    # 2. 均线
    # =========================================================

    for period in [5, 10, 20, 30, 60, 120]:
        ma = close.rolling(
            period,
            min_periods=period,
        ).mean()

        data[f"MA_{period}"] = ma

        data[f"PRICE_MA_{period}"] = (
            _safe_div(close, ma) - 1
        )

    # =========================================================
    # 3. 波动率
    # =========================================================

    daily_return = close.pct_change()

    for period in [5, 10, 20, 60]:
        data[f"STD_{period}"] = (
            daily_return
            .rolling(
                period,
                min_periods=period,
            )
            .std()
        )

    # =========================================================
    # 4. 成交量特征
    # =========================================================

    for period in [5, 10, 20, 60]:
        volume_ma = volume.rolling(
            period,
            min_periods=period,
        ).mean()

        data[f"VOL_MA_{period}"] = volume_ma

        data[f"VOL_RATIO_{period}"] = _safe_div(
            volume,
            volume_ma,
        )

    # =========================================================
    # 5. 价格位置
    # =========================================================

    for period in [20, 60, 120]:
        rolling_high = high.rolling(
            period,
            min_periods=period,
        ).max()

        rolling_low = low.rolling(
            period,
            min_periods=period,
        ).min()

        data[f"HIGH_POS_{period}"] = _safe_div(
            close - rolling_low,
            rolling_high - rolling_low,
        )

        data[f"DIST_HIGH_{period}"] = (
            _safe_div(
                close,
                rolling_high,
            ) - 1
        )

        data[f"DIST_LOW_{period}"] = (
            _safe_div(
                close,
                rolling_low,
            ) - 1
        )

    # =========================================================
    # 6. K线结构
    # =========================================================

    price_range = high - low

    data["BAR_RANGE"] = _safe_div(
        price_range,
        close,
    )

    data["BODY"] = _safe_div(
        close - open_price,
        open_price,
    )

    data["BODY_ABS"] = _safe_div(
        (close - open_price).abs(),
        open_price,
    )

    data["UPPER_SHADOW"] = _safe_div(
        high - pd.concat(
            [open_price, close],
            axis=1,
        ).max(axis=1),
        close,
    )

    data["LOWER_SHADOW"] = _safe_div(
        pd.concat(
            [open_price, close],
            axis=1,
        ).min(axis=1) - low,
        close,
    )

    # =========================================================
    # 7. RSI
    # =========================================================

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(
        14,
        min_periods=14,
    ).mean()

    avg_loss = loss.rolling(
        14,
        min_periods=14,
    ).mean()

    rs = _safe_div(
        avg_gain,
        avg_loss,
    )

    data["RSI_14"] = (
        100 - (100 / (1 + rs))
    )

    # =========================================================
    # 8. ATR 风格波动特征
    # =========================================================

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr14 = true_range.rolling(
        14,
        min_periods=14,
    ).mean()

    data["ATR_14"] = _safe_div(
        atr14,
        close,
    )

    # =========================================================
    # 9. 动量
    # =========================================================

    for short_period, long_period in [
        (5, 20),
        (10, 20),
        (20, 60),
        (60, 120),
    ]:
        short_ma = close.rolling(
            short_period,
            min_periods=short_period,
        ).mean()

        long_ma = close.rolling(
            long_period,
            min_periods=long_period,
        ).mean()

        data[
            f"MA_MOM_{short_period}_{long_period}"
        ] = _safe_div(
            short_ma,
            long_ma,
        ) - 1

    # =========================================================
    # 10. 成交额 / 换手率
    # =========================================================

    if "turnover" in data.columns:
        turnover = data["turnover"]

        turnover_ma20 = turnover.rolling(
            20,
            min_periods=20,
        ).mean()

        data["TURNOVER_MA20"] = turnover_ma20

        data["TURNOVER_RATIO"] = _safe_div(
            turnover,
            turnover_ma20,
        )

    # =========================================================
    # 11. 缺失值处理
    # =========================================================

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return data
