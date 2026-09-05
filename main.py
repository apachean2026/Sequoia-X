"""Sequoia-X V2 主程序入口。

两种运行模式：
python main.py                # 日常模式：增量补数据 + 跑策略 + 飞书推送
python main.py --backfill     # 回填模式：baostock 拉全市场历史K线
"""

import argparse
import sys
from datetime import date

import socket
from dotenv import load_dotenv

load_dotenv()

socket.setdefaulttimeout(10.0)

from sequoia_x.core.config import get_settings
from sequoia_x.core.logger import get_logger
from sequoia_x.data.engine import DataEngine
from sequoia_x.notify.feishu import FeishuNotifier
from sequoia_x.strategy.base import BaseStrategy
from sequoia_x.strategy.high_tight_flag import HighTightFlagStrategy
from sequoia_x.strategy.limit_up_shakeout import LimitUpShakeoutStrategy
from sequoia_x.strategy.ma_volume import MaVolumeStrategy
from sequoia_x.strategy.turtle_trade import TurtleTradeStrategy
from sequoia_x.strategy.uptrend_limit_down import UptrendLimitDownStrategy
from sequoia_x.strategy.rps_breakout import RpsBreakoutStrategy
from sequoia_x.strategy.private_placement import PrivatePlacementStrategy


# ============================================================
# 策略中文名称映射
# ============================================================

STRATEGY_NAMES = {
    "MaVolumeStrategy": "均线放量突破",
    "TurtleTradeStrategy": "海龟趋势策略",
    "HighTightFlagStrategy": "高紧旗形突破",
    "LimitUpShakeoutStrategy": "涨停洗盘策略",
    "UptrendLimitDownStrategy": "上涨回调策略",
    "RpsBreakoutStrategy": "RPS强势突破",
    "PrivatePlacementStrategy": "定增公告策略",
}


def get_strategy_display_name(strategy: BaseStrategy) -> str:
    """获取策略对外显示名称。

    如果策略已经配置中文名称，就显示中文；
    如果没有配置，则继续显示原来的 Python 类名。
    """

    class_name = type(strategy).__name__

    return STRATEGY_NAMES.get(
        class_name,
        class_name,
    )


def main() -> None:
    """Sequoia-X V2 主流程。"""

    parser = argparse.ArgumentParser(
        description="Sequoia-X V2 选股系统"
    )

    parser.add_argument(
        "--backfill",
        action="store_true",
        help="回填模式：通过 baostock 拉取全市场历史 K 线",
    )

    args = parser.parse_args()

    logger = None

    try:
        # ====================================================
        # 1. 初始化配置
        # ====================================================

        settings = get_settings()

        # ====================================================
        # 2. 初始化日志
        # ====================================================

        logger = get_logger(__name__)

        logger.info("Sequoia-X V2 启动")

        # ====================================================
        # 3. 初始化数据引擎
        # ====================================================

        engine = DataEngine(settings)

        # ====================================================
        # 回填模式
        # ====================================================

        if args.backfill:
            logger.info("进入回填模式...")

            all_symbols = engine.get_all_symbols()

            logger.info(
                f"准备回填 {len(all_symbols)} 只股票"
            )

            engine.backfill(all_symbols)

            logger.info(
                "Sequoia-X V2 回填模式运行完成"
            )

            return

        # ====================================================
        # 日常模式
        # ====================================================

        logger.info("开始拉取最新快照...")

        count = engine.sync_today_bulk()

        logger.info(
            f"快照同步完成，写入 {count} 只股票"
        )

        # ====================================================
        # 4. 策略列表
        # ====================================================

        strategies: list[BaseStrategy] = [
            MaVolumeStrategy(
                engine=engine,
                settings=settings,
            ),

            TurtleTradeStrategy(
                engine=engine,
                settings=settings,
            ),

            HighTightFlagStrategy(
                engine=engine,
                settings=settings,
            ),

            LimitUpShakeoutStrategy(
                engine=engine,
                settings=settings,
            ),

            UptrendLimitDownStrategy(
                engine=engine,
                settings=settings,
            ),

            RpsBreakoutStrategy(
                engine=engine,
                settings=settings,
            ),

            PrivatePlacementStrategy(
                engine=engine,
                settings=settings,
            ),
        ]

        # ====================================================
        # 5. 初始化飞书通知
        # ====================================================

        notifier = FeishuNotifier(settings)

        # ====================================================
        # 6. 执行所有策略
        # ====================================================

        for strategy in strategies:

            # 获取中文策略名称
            strategy_name = get_strategy_display_name(strategy)

            # 获取 Python 内部类名
            class_name = type(strategy).__name__

            logger.info(
                f"执行策略：{strategy_name}"
            )

            logger.info(
                f"策略内部名称：{class_name}"
            )

            # 执行策略
            selected: list[str] = strategy.run()

            logger.info(
                f"{strategy_name} 选出 {len(selected)} 只股票"
            )

            # =================================================
            # 有选股结果
            # =================================================

            if selected:

                notifier.send(
                    symbols=selected,
                    strategy_name=strategy_name,
                    webhook_key=strategy.webhook_key,
                )

            # =================================================
            # 没有选股结果
            # =================================================

            else:

                logger.info(
                    f"{strategy_name} 无选股结果，跳过推送"
                )

        logger.info(
            "Sequoia-X V2 运行完成"
        )

    except Exception:

        if logger is not None:

            logger.exception(
                "主流程发生未捕获异常，程序终止"
            )

        else:

            import traceback

            traceback.print_exc()

        sys.exit(1)


if __name__ == "__main__":
    main()
