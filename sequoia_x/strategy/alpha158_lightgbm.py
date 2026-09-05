"""Alpha158 + LightGBM AI选股策略。

第一阶段：
    先验证 AI 策略能够正常接入 Sequoia-X。

后续阶段：
    1. Alpha158 特征工程
    2. LightGBM 训练
    3. 未来收益预测
    4. TopK 选股
"""

from __future__ import annotations

import logging

import pandas as pd

from sequoia_x.strategy.base import BaseStrategy


logger = logging.getLogger(__name__)


class Alpha158LightGBMStrategy(BaseStrategy):
    """Alpha158 + LightGBM AI选股策略。"""

    # 飞书机器人路由
    # 第一阶段先使用默认机器人。
    webhook_key = "default"

    # 对外显示名称
    display_name = "AI量化：Alpha158 + LightGBM"

    def run(self) -> list[str]:
        """执行 AI 选股。

        第一阶段暂时只做数据库读取测试。
        真正的 Alpha158 + LightGBM 会在后续阶段加入。
        """

        logger.info(
            "开始执行 Alpha158 + LightGBM AI策略"
        )

        # ------------------------------------------------------
        # 获取本地全部股票
        # ------------------------------------------------------

        symbols = self.engine.get_local_symbols()

        if not symbols:
            logger.warning(
                "本地数据库没有股票数据，"
                "Alpha158 AI策略跳过"
            )

            return []

        logger.info(
            f"AI策略读取到 {len(symbols)} 只股票"
        )

        # ------------------------------------------------------
        # 第一阶段：
        # 检查数据是否完整
        # ------------------------------------------------------

        valid_symbols: list[str] = []

        for symbol in symbols:

            try:

                df = self.engine.get_ohlcv(
                    symbol
                )

                if df.empty:
                    continue

                # 至少需要足够的历史数据
                # 后续 Alpha158 会需要更多历史窗口。
                if len(df) < 120:
                    continue

                # 检查必要字段
                required_columns = {
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                }

                if not required_columns.issubset(
                    df.columns
                ):
                    continue

                # 转换日期
                df["date"] = pd.to_datetime(
                    df["date"],
                    errors="coerce",
                )

                # 删除无效数据
                df = df.dropna(
                    subset=[
                        "date",
                        "close",
                        "volume",
                    ]
                )

                if len(df) < 120:
                    continue

                valid_symbols.append(
                    symbol
                )

            except Exception as exc:

                logger.warning(
                    f"[{symbol}] "
                    f"AI数据检查失败：{exc}"
                )

        # ------------------------------------------------------
        # 输出检查结果
        # ------------------------------------------------------

        logger.info(
            "Alpha158 AI策略数据检查完成："
            f"{len(valid_symbols)} / "
            f"{len(symbols)} 只股票可用"
        )

        # ------------------------------------------------------
        # 第一阶段暂不真正选股
        # ------------------------------------------------------

        logger.info(
            "Alpha158 + LightGBM "
            "第一阶段数据检查完成，"
            "暂不产生实际选股结果"
        )

        return []
