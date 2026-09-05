"""Sequoia-X V2 主程序入口。

两种运行模式：
python main.py               # 日常模式：8进程增量补数据 + 跑策略 + 飞书推送（2~3分钟）
python main.py --backfill    # 回填模式：baostock 拉全市场历史K线（首次/补数据用，约12分钟）
"""

import argparse
import sys
from dotenv import load_dotenv

load_dotenv()

from datetime import date

import socket

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

# 左边：Python 程序内部使用的策略类名

# 右边：飞书通知、日志中显示的中文名称

#

# 注意：

# 不要修改左边的英文类名，否则可能导致策略无法正常运行。

# 以后新增策略，只需要在这里增加一行即可。

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
"""
获取策略对外显示名称。

```
如果策略已经配置中文名称，就显示中文；
如果没有配置，则继续显示原来的 Python 类名。
"""

class_name = type(strategy).__name__

return STRATEGY_NAMES.get(
    class_name,
    class_name,
)
```

def main() -> None:
parser = argparse.ArgumentParser(
description="Sequoia-X V2 选股系统"
)

```
parser.add_argument(
    "--backfill",
    action="store_true",
    help="回填模式：通过 baostock 拉取全市场历史 K 线（约12分钟）",
)

args = parser.parse_args()

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

    if args.backfill:
        # ── 回填模式：单线程保守拉历史 K 线，自动多轮重跑 ──

        logger.info("进入回填模式...")

        all_symbols = engine.get_all_symbols()

        engine.backfill(all_symbols)

        logger.info(
            "Sequoia-X V2 回填模式运行完成"
        )

        return

    # ====================================================
    # 日常模式
    # ====================================================
    # 单次 API 补今天 + 策略 + 推送
    # ====================================================

    logger.info("开始拉取最新快照...")

    count = engine.sync_today_bulk()

    logger.info(
        f"快照同步完成，写入 {count} 只股票"
    )

    # ====================================================
    # 4. 策略列表
    # ====================================================
    # 新增策略时，在这里追加即可。
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
    # 6. 遍历策略
    # ====================================================

    for strategy in strategies:

        # ------------------------------------------------
        # 获取中文策略名称
        # ------------------------------------------------

        strategy_name = get_strategy_display_name(strategy)

        # ------------------------------------------------
        # 获取程序内部英文类名
        # ------------------------------------------------

        class_name = type(strategy).__name__

        # ------------------------------------------------
        # 日志显示中文名称
        # ------------------------------------------------

        logger.info(
            f"执行策略：{strategy_name}"
        )

        # ------------------------------------------------
        # 执行策略
        # ------------------------------------------------

        selected: list[str] = strategy.run()

        # ------------------------------------------------
        # 日志显示中文名称
        # ------------------------------------------------

        logger.info(
            f"{strategy_name} 选出 {len(selected)} 只股票"
        )

        # ------------------------------------------------
        # 有选股结果
        # ------------------------------------------------

        if selected:

            notifier.send(
                symbols=selected,

                # 这里传中文名称
                # 所以飞书通知会显示中文策略名
                strategy_name=strategy_name,

                # webhook_key 仍然使用策略自己的配置
                # 不要修改
                webhook_key=strategy.webhook_key,
            )

        # ------------------------------------------------
        # 没有选股结果
        # ------------------------------------------------

        else:

            logger.info(
                f"{strategy_name} 无选股结果，跳过推送"
            )

except Exception:

    try:

        _logger = get_logger(__name__)

        _logger.exception(
            "主流程发生未捕获异常，程序终止"
        )

    except Exception:

        import traceback

        traceback.print_exc()

    sys.exit(1)

logger.info(
    "Sequoia-X V2 运行完成"
)
```

if **name** == "**main**":
main()


if __name__ == "__main__":
    main()
