"""Alpha158 风格特征工程。

第一阶段目标：
    从 Sequoia-X 现有 stock_daily 数据中，
    计算适合 LightGBM 使用的量价特征。

说明：
    这里不依赖 Qlib，直接使用 Sequoia-X 已有的 OHLCV 数据。

注意：
    本模块只负责特征工程，不负责训练、不负责预测、
    不负责回测。

    特征全部基于当前及历史数据计算，
    不使用未来数据，避免数据泄漏。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# =========================================================
# 基础工具
# =========================================================

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """安全除法，避免除以 0。"""
    denominator = b.replace(0, np.nan)
    return a / denominator


def _rolling_mean(
    series: pd.Series,
    period: int,
) -> pd.Series:
    """滚动均值。"""
    return series.rolling(
        period,
        min_periods=period,
    ).mean()


def _rolling_std(
    series: pd.Series,
    period: int,
) -> pd.Series:
    """滚动标准差。"""
    return series.rolling(
        period,
        min_periods=period,
    ).std()


def _rolling_max(
    series: pd.Series,
    period: int,
) -> pd.Series:
    """滚动最大值。"""
    return series.rolling(
        period,
        min_periods=period,
    ).max()


def _rolling_min(
    series: pd.Series,
    period: int,
) -> pd.Series:
    """滚动最小值。"""
    return series.rolling(
        period,
        min_periods=period,
    ).min()


# =========================================================
# 主特征工程
# =========================================================

def build_alpha158_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
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
        原始行情 + Alpha158 风格特征。

    注意：
        所有特征只使用当前及历史数据，
        不使用未来数据。
    """

    if df.empty:
        return pd.DataFrame()

    data = df.copy()

    # =========================================================
    # 0. 字段检查
    # =========================================================

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

    # =========================================================
    # 1. 数据类型标准化
    # =========================================================

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

    data = (
        data
        .dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates(subset=["date"])
        .reset_index(drop=True)
    )

    if data.empty:
        return data

    # =========================================================
    # 2. 基础价格变量
    # =========================================================

    open_price = data["open"]
    high = data["high"]
    low = data["low"]
    close = data["close"]
    volume = data["volume"]

    previous_close = close.shift(1)

    # =========================================================
    # 3. 收益率 / 动量
    # =========================================================

    for period in [
        1,
        2,
        3,
        5,
        10,
        20,
        30,
        60,
        120,
    ]:
        data[f"RET_{period}"] = (
            _safe_div(
                close,
                close.shift(period),
            ) - 1
        )

    # 日收益率
    daily_return = close.pct_change()

    # 对数收益率
    data["LOG_RET_1"] = np.log(
        _safe_div(
            close,
            previous_close,
        )
    )

    # =========================================================
    # 4. 均线 / 价格相对均线
    # =========================================================

    for period in [
        5,
        10,
        20,
        30,
        60,
        120,
    ]:
        ma = _rolling_mean(
            close,
            period,
        )

        data[f"MA_{period}"] = ma

        data[f"PRICE_MA_{period}"] = (
            _safe_div(
                close,
                ma,
            ) - 1
        )

        # 均线斜率
        data[f"MA_SLOPE_{period}"] = (
            _safe_div(
                ma,
                ma.shift(period),
            ) - 1
        )

    # =========================================================
    # 5. 均线之间的趋势关系
    # =========================================================

    ma5 = _rolling_mean(close, 5)
    ma10 = _rolling_mean(close, 10)
    ma20 = _rolling_mean(close, 20)
    ma30 = _rolling_mean(close, 30)
    ma60 = _rolling_mean(close, 60)
    ma120 = _rolling_mean(close, 120)

    data["MA_5_10"] = _safe_div(
        ma5,
        ma10,
    ) - 1

    data["MA_5_20"] = _safe_div(
        ma5,
        ma20,
    ) - 1

    data["MA_10_20"] = _safe_div(
        ma10,
        ma20,
    ) - 1

    data["MA_20_60"] = _safe_div(
        ma20,
        ma60,
    ) - 1

    data["MA_30_60"] = _safe_div(
        ma30,
        ma60,
    ) - 1

    data["MA_60_120"] = _safe_div(
        ma60,
        ma120,
    ) - 1

    # =========================================================
    # 6. 波动率
    # =========================================================

    for period in [
        5,
        10,
        20,
        30,
        60,
    ]:
        data[f"STD_{period}"] = _rolling_std(
            daily_return,
            period,
        )

        # 年化风格波动率
        data[f"VOLATILITY_{period}"] = (
            data[f"STD_{period}"]
            * np.sqrt(252)
        )

    # 波动率变化
    data["STD_5_20"] = _safe_div(
        data["STD_5"],
        data["STD_20"],
    )

    data["STD_10_60"] = _safe_div(
        data["STD_10"],
        data["STD_60"],
    )

    # =========================================================
    # 7. 最高价 / 最低价 / 价格位置
    # =========================================================

    for period in [
        5,
        10,
        20,
        30,
        60,
        120,
    ]:
        rolling_high = _rolling_max(
            high,
            period,
        )

        rolling_low = _rolling_min(
            low,
            period,
        )

        price_range = (
            rolling_high - rolling_low
        )

        data[f"HIGH_POS_{period}"] = _safe_div(
            close - rolling_low,
            price_range,
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
    # 8. 突破 / 新高 / 新低
    # =========================================================

    for period in [
        5,
        10,
        20,
        60,
    ]:
        previous_high = (
            high
            .shift(1)
            .rolling(
                period,
                min_periods=period,
            )
            .max()
        )

        previous_low = (
            low
            .shift(1)
            .rolling(
                period,
                min_periods=period,
            )
            .min()
        )

        data[f"BREAK_HIGH_{period}"] = (
            _safe_div(
                close,
                previous_high,
            ) - 1
        )

        data[f"BREAK_LOW_{period}"] = (
            _safe_div(
                close,
                previous_low,
            ) - 1
        )

    # =========================================================
    # 9. K 线结构
    # =========================================================

    price_range = high - low

    body = close - open_price

    data["BAR_RANGE"] = _safe_div(
        price_range,
        close,
    )

    data["BODY"] = _safe_div(
        body,
        open_price,
    )

    data["BODY_ABS"] = _safe_div(
        body.abs(),
        open_price,
    )

    data["BODY_RANGE_RATIO"] = _safe_div(
        body.abs(),
        price_range,
    )

    candle_max = pd.concat(
        [
            open_price,
            close,
        ],
        axis=1,
    ).max(axis=1)

    candle_min = pd.concat(
        [
            open_price,
            close,
        ],
        axis=1,
    ).min(axis=1)

    data["UPPER_SHADOW"] = _safe_div(
        high - candle_max,
        close,
    )

    data["LOWER_SHADOW"] = _safe_div(
        candle_min - low,
        close,
    )

    data["CLOSE_POSITION"] = _safe_div(
        close - low,
        price_range,
    )

    data["OPEN_POSITION"] = _safe_div(
        open_price - low,
        price_range,
    )

    # =========================================================
    # 10. K 线方向
    # =========================================================

    data["UP_DAY"] = (
        close > previous_close
    ).astype(float)

    data["DOWN_DAY"] = (
        close < previous_close
    ).astype(float)

    data["OPEN_GAP"] = (
        _safe_div(
            open_price,
            previous_close,
        ) - 1
    )

    data["HIGH_CHANGE"] = (
        _safe_div(
            high,
            previous_close,
        ) - 1
    )

    data["LOW_CHANGE"] = (
        _safe_div(
            low,
            previous_close,
        ) - 1
    )

    # =========================================================
    # 11. RSI
    # =========================================================

    delta = close.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    for period in [
        6,
        14,
        24,
    ]:
        avg_gain = gain.rolling(
            period,
            min_periods=period,
        ).mean()

        avg_loss = loss.rolling(
            period,
            min_periods=period,
        ).mean()

        rs = _safe_div(
            avg_gain,
            avg_loss,
        )

        data[f"RSI_{period}"] = (
            100
            - (
                100
                / (1 + rs)
            )
        )

    # =========================================================
    # 12. ATR / 真实波动幅度
    # =========================================================

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    for period in [
        5,
        14,
        20,
    ]:
        atr = true_range.rolling(
            period,
            min_periods=period,
        ).mean()

        data[f"ATR_{period}"] = _safe_div(
            atr,
            close,
        )

    # =========================================================
    # 13. 动量 / 均线动量
    # =========================================================

    for short_period, long_period in [
        (3, 10),
        (5, 20),
        (10, 20),
        (10, 30),
        (20, 60),
        (30, 60),
        (60, 120),
    ]:
        short_ma = _rolling_mean(
            close,
            short_period,
        )

        long_ma = _rolling_mean(
            close,
            long_period,
        )

        data[
            f"MA_MOM_{short_period}_{long_period}"
        ] = (
            _safe_div(
                short_ma,
                long_ma,
            ) - 1
        )

    # =========================================================
    # 14. 成交量特征
    # =========================================================

    for period in [
        5,
        10,
        20,
        30,
        60,
    ]:
        volume_ma = _rolling_mean(
            volume,
            period,
        )

        data[f"VOL_MA_{period}"] = (
            volume_ma
        )

        data[f"VOL_RATIO_{period}"] = (
            _safe_div(
                volume,
                volume_ma,
            )
        )

        data[f"VOL_STD_{period}"] = (
            _rolling_std(
                volume,
                period,
            )
        )

    # =========================================================
    # 15. 成交量变化
    # =========================================================

    data["VOL_CHANGE_1"] = (
        _safe_div(
            volume,
            volume.shift(1),
        ) - 1
    )

    data["VOL_CHANGE_5"] = (
        _safe_div(
            volume,
            volume.shift(5),
        ) - 1
    )

    data["VOL_CHANGE_20"] = (
        _safe_div(
            volume,
            volume.shift(20),
        ) - 1
    )

    # =========================================================
    # 16. 量价关系
    # =========================================================

    for period in [
        5,
        10,
        20,
        60,
    ]:
        price_ret = (
            _safe_div(
                close,
                close.shift(period),
            ) - 1
        )

        volume_ret = (
            _safe_div(
                volume,
                volume.shift(period),
            ) - 1
        )

        data[
            f"PRICE_VOLUME_MOM_{period}"
        ] = (
            price_ret * volume_ret
        )

    # =========================================================
    # 17. OBV 风格特征
    # =========================================================

    direction = np.sign(
        close.diff()
    ).fillna(0)

    obv_change = (
        volume * direction
    )

    obv = obv_change.cumsum()

    data["OBV"] = obv

    for period in [
        5,
        20,
        60,
    ]:
        obv_ma = _rolling_mean(
            obv,
            period,
        )

        data[f"OBV_RATIO_{period}"] = (
            _safe_div(
                obv,
                obv_ma,
            ) - 1
        )

    # =========================================================
    # 18. VWAP 风格特征
    # =========================================================

    typical_price = (
        high + low + close
    ) / 3

    pv = (
        typical_price * volume
    )

    for period in [
        5,
        20,
        60,
    ]:
        rolling_pv = pv.rolling(
            period,
            min_periods=period,
        ).sum()

        rolling_volume = volume.rolling(
            period,
            min_periods=period,
        ).sum()

        vwap = _safe_div(
            rolling_pv,
            rolling_volume,
        )

        data[f"VWAP_{period}"] = vwap

        data[f"PRICE_VWAP_{period}"] = (
            _safe_div(
                close,
                vwap,
            ) - 1
        )

    # =========================================================
    # 19. 成交量趋势
    # =========================================================

    volume_ma5 = _rolling_mean(
        volume,
        5,
    )

    volume_ma20 = _rolling_mean(
        volume,
        20,
    )

    volume_ma60 = _rolling_mean(
        volume,
        60,
    )

    data["VOL_MA_5_20"] = (
        _safe_div(
            volume_ma5,
            volume_ma20,
        ) - 1
    )

    data["VOL_MA_20_60"] = (
        _safe_div(
            volume_ma20,
            volume_ma60,
        ) - 1
    )

    # =========================================================
    # 20. 连续上涨 / 下跌
    # =========================================================

    up = (
        daily_return > 0
    ).astype(int)

    down = (
        daily_return < 0
    ).astype(int)

    data["UP_COUNT_5"] = (
        up.rolling(
            5,
            min_periods=5,
        ).sum()
    )

    data["UP_COUNT_10"] = (
        up.rolling(
            10,
            min_periods=10,
        ).sum()
    )

    data["UP_COUNT_20"] = (
        up.rolling(
            20,
            min_periods=20,
        ).sum()
    )

    data["DOWN_COUNT_5"] = (
        down.rolling(
            5,
            min_periods=5,
        ).sum()
    )

    data["DOWN_COUNT_10"] = (
        down.rolling(
            10,
            min_periods=10,
        ).sum()
    )

    data["DOWN_COUNT_20"] = (
        down.rolling(
            20,
            min_periods=20,
        ).sum()
    )

    # =========================================================
    # 21. 收益率统计
    # =========================================================

    for period in [
        5,
        10,
        20,
        60,
    ]:
        data[f"RET_MEAN_{period}"] = (
            daily_return.rolling(
                period,
                min_periods=period,
            ).mean()
        )

        data[f"RET_MAX_{period}"] = (
            daily_return.rolling(
                period,
                min_periods=period,
            ).max()
        )

        data[f"RET_MIN_{period}"] = (
            daily_return.rolling(
                period,
                min_periods=period,
            ).min()
        )

    # =========================================================
    # 22. 偏度 / 峰度风格特征
    # =========================================================

    for period in [
        20,
        60,
    ]:
        data[f"RET_SKEW_{period}"] = (
            daily_return.rolling(
                period,
                min_periods=period,
            ).skew()
        )

        data[f"RET_KURT_{period}"] = (
            daily_return.rolling(
                period,
                min_periods=period,
            ).kurt()
        )

    # =========================================================
    # 23. Turnover / 换手率
    # =========================================================

    if "turnover" in data.columns:
        turnover = data["turnover"]

        for period in [
            5,
            10,
            20,
            60,
        ]:
            turnover_ma = _rolling_mean(
                turnover,
                period,
            )

            data[
                f"TURNOVER_MA{period}"
            ] = turnover_ma

            data[
                f"TURNOVER_RATIO_{period}"
            ] = _safe_div(
                turnover,
                turnover_ma,
            )

        data["TURNOVER_CHANGE_1"] = (
            _safe_div(
                turnover,
                turnover.shift(1),
            ) - 1
        )

    # =========================================================
    # 24. 价格与成交量相关性
    # =========================================================

    for period in [
        10,
        20,
        60,
    ]:
        data[
            f"PRICE_VOLUME_CORR_{period}"
        ] = (
            close
            .rolling(
                period,
                min_periods=period,
            )
            .corr(volume)
        )

    # =========================================================
    # 25. 收益率与成交量相关性
    # =========================================================

    for period in [
        10,
        20,
        60,
    ]:
        data[
            f"RET_VOLUME_CORR_{period}"
        ] = (
            daily_return
            .rolling(
                period,
                min_periods=period,
            )
            .corr(volume)
        )

    # =========================================================
    # 26. 波动压缩 / 波动扩张
    # =========================================================

    std5 = _rolling_std(
        daily_return,
        5,
    )

    std20 = _rolling_std(
        daily_return,
        20,
    )

    std60 = _rolling_std(
        daily_return,
        60,
    )

    data["VOL_COMPRESSION_5_20"] = (
        _safe_div(
            std5,
            std20,
        )
    )

    data["VOL_COMPRESSION_20_60"] = (
        _safe_div(
            std20,
            std60,
        )
    )

    # =========================================================
    # 27. 价格通道宽度
    # =========================================================

    for period in [
        20,
        60,
        120,
    ]:
        rolling_high = _rolling_max(
            high,
            period,
        )

        rolling_low = _rolling_min(
            low,
            period,
        )

        data[
            f"CHANNEL_WIDTH_{period}"
        ] = _safe_div(
            rolling_high - rolling_low,
            close,
        )

    # =========================================================
    # 28. 收盘价相对历史区间
    # =========================================================

    for period in [
        20,
        60,
        120,
    ]:
        rolling_high = _rolling_max(
            high,
            period,
        )

        rolling_low = _rolling_min(
            low,
            period,
        )

        midpoint = (
            rolling_high
            + rolling_low
        ) / 2

        data[
            f"PRICE_MIDPOINT_DIST_{period}"
        ] = _safe_div(
            close - midpoint,
            close,
        )

    # =========================================================
    # 29. 价格加速度
    # =========================================================

    ret5 = (
        _safe_div(
            close,
            close.shift(5),
        ) - 1
    )

    ret10 = (
        _safe_div(
            close,
            close.shift(10),
        ) - 1
    )

    ret20 = (
        _safe_div(
            close,
            close.shift(20),
        ) - 1
    )

    data["MOM_ACCEL_5_10"] = (
        ret5 - ret10
    )

    data["MOM_ACCEL_10_20"] = (
        ret10 - ret20
    )

    # =========================================================
    # 30. 清理异常值
    # =========================================================

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # =========================================================
    # 31. 保证日期排序
    # =========================================================

    data = (
        data
        .sort_values("date")
        .reset_index(drop=True)
    )

    return data
