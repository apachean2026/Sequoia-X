"""数据引擎模块：负责 SQLite 行情数据存储与 baostock 增量同步。"""

import sqlite3
from pathlib import Path

import pandas as pd

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS stock_daily (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol   TEXT    NOT NULL,
    date     TEXT    NOT NULL,
    open     REAL,
    high     REAL,
    low      REAL,
    close    REAL,
    volume   REAL,
    turnover REAL,
    UNIQUE (symbol, date)
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_symbol_date ON stock_daily (symbol, date);
"""


class DataEngine:
    """行情数据引擎，负责 SQLite 存储和 baostock 数据同步。"""

    def __init__(self, settings: Settings) -> None:
        self.db_path: str = settings.db_path
        self.start_date: str = settings.start_date
        self._init_db()

    def _init_db(self) -> None:
        """初始化 SQLite 数据库和数据表。"""

        Path(self.db_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(_CREATE_INDEX_SQL)
            conn.commit()

        logger.info(
            f"数据库初始化完成：{self.db_path}"
        )

    def _get_last_date(
        self,
        symbol: str,
    ) -> str | None:
        """获取某只股票本地数据库中的最新日期。"""

        with sqlite3.connect(
            self.db_path
        ) as conn:

            row = conn.execute(
                """
                SELECT MAX(date)
                FROM stock_daily
                WHERE symbol = ?
                """,
                (symbol,),
            ).fetchone()

        return (
            row[0]
            if row and row[0]
            else None
        )

    def get_ohlcv(
        self,
        symbol: str,
    ) -> pd.DataFrame:
        """读取某只股票的全部日 K 数据。"""

        with sqlite3.connect(
            self.db_path
        ) as conn:

            df = pd.read_sql(
                """
                SELECT *
                FROM stock_daily
                WHERE symbol = ?
                ORDER BY date
                """,
                conn,
                params=(symbol,),
            )

        return df

    @staticmethod
    def _to_baostock_code(
        symbol: str,
    ) -> str:
        """将纯数字代码转换为 baostock 格式。

        6/9 开头 -> sh
        其余 -> sz
        """

        prefix = (
            "sh"
            if symbol.startswith(("6", "9"))
            else "sz"
        )

        return f"{prefix}.{symbol}"

    # ============================================================
    # 每日行情增量同步
    # ============================================================

    def sync_today_bulk(self) -> int:
        """稳定模式同步最新行情。

        主要特点：

        1. 不使用多进程。
        2. 使用单个 BaoStock 连接。
        3. 每只股票失败自动重试。
        4. 查询失败自动重新登录。
        5. 每 100 只股票主动重连。
        6. 每次请求之间增加间隔。
        7. 单只股票失败不会导致整个任务终止。
        8. 周六、周日自动跳过。
        9. 使用 INSERT OR REPLACE 写入 SQLite。
        """

        import time
        from datetime import date, timedelta

        import baostock as bs

        today = date.today()

        # --------------------------------------------------------
        # 周六、周日跳过
        # --------------------------------------------------------

        if today.weekday() >= 5:

            logger.info(
                f"今天是非交易日 {today}，"
                f"跳过行情同步"
            )

            return 0

        today_str = today.strftime(
            "%Y-%m-%d"
        )

        # --------------------------------------------------------
        # 获取本地股票以及最后日期
        # --------------------------------------------------------

        with sqlite3.connect(
            self.db_path
        ) as conn:

            rows = conn.execute(
                """
                SELECT symbol, MAX(date)
                FROM stock_daily
                GROUP BY symbol
                """
            ).fetchall()

        if not rows:

            logger.warning(
                "本地无股票数据，"
                "请先执行 --backfill"
            )

            return 0

        # --------------------------------------------------------
        # 创建待更新股票列表
        # --------------------------------------------------------

        tasks = []

        for symbol, last_date in rows:

            # 已经有今天的数据
            if (
                last_date
                and last_date >= today_str
            ):
                continue

            # 默认从今天开始
            start = today_str

            # 如果本地存在历史数据，
            # 则从最后一天的下一天开始
            if last_date:

                start = (
                    date.fromisoformat(
                        last_date
                    )
                    + timedelta(days=1)
                ).strftime(
                    "%Y-%m-%d"
                )

            tasks.append(
                (
                    symbol,
                    self._to_baostock_code(
                        symbol
                    ),
                    start,
                    today_str,
                )
            )

        if not tasks:

            logger.info(
                "所有股票已是最新，"
                "无需更新"
            )

            return 0

        logger.info(
            f"需要更新 {len(tasks)} 只股票，"
            f"启动稳定模式同步："
            f"单连接 + 自动重试 + 自动重连"
        )

        # ========================================================
        # BaoStock 登录函数
        # ========================================================

        def login_baostock() -> bool:
            """登录 BaoStock。"""

            try:

                lg = bs.login()

                if lg.error_code != "0":

                    logger.error(
                        f"baostock 登录失败："
                        f"{lg.error_msg}"
                    )

                    return False

                logger.info(
                    "baostock 登录成功"
                )

                return True

            except Exception as exc:

                logger.error(
                    f"baostock 登录异常："
                    f"{exc}"
                )

                return False

        # --------------------------------------------------------
        # 第一次登录
        # --------------------------------------------------------

        if not login_baostock():

            logger.error(
                "无法连接 baostock，"
                "终止本次行情同步"
            )

            return 0

        # ========================================================
        # 同步参数
        # ========================================================

        all_rows = []

        success = 0
        empty = 0
        failed = 0

        # 每 100 只股票主动重连
        reconnect_every = 100

        # 每次请求之间暂停
        request_interval = 0.15

        # 每只股票最大尝试次数
        max_retries = 3

        try:

            # ====================================================
            # 开始逐只股票同步
            # ====================================================

            for index, (
                symbol,
                bs_code,
                start,
                end,
            ) in enumerate(
                tasks,
                1,
            ):

                # ------------------------------------------------
                # 每 100 只股票主动重连
                # ------------------------------------------------

                if (
                    index > 1
                    and
                    (index - 1)
                    % reconnect_every
                    == 0
                ):

                    logger.info(
                        f"已处理 "
                        f"{index - 1}/"
                        f"{len(tasks)}，"
                        f"主动重连 baostock..."
                    )

                    try:
                        bs.logout()
                    except Exception:
                        pass

                    time.sleep(2)

                    # 第一次重连
                    if not login_baostock():

                        logger.warning(
                            "baostock 重连失败，"
                            "5 秒后再次尝试..."
                        )

                        time.sleep(5)

                        # 第二次重连
                        if not login_baostock():

                            logger.error(
                                "baostock 连续重连失败，"
                                "停止本次行情同步"
                            )

                            break

                # ------------------------------------------------
                # 查询当前股票
                # ------------------------------------------------

                query_ok = False
                rows_for_stock = []

                for attempt in range(
                    1,
                    max_retries + 1,
                ):

                    try:

                        rs = (
                            bs.query_history_k_data_plus(
                                bs_code,
                                (
                                    "date,open,high,low,"
                                    "close,volume,amount"
                                ),
                                start_date=start,
                                end_date=end,
                                frequency="d",
                                adjustflag="1",
                            )
                        )

                        # BaoStock 返回错误
                        if (
                            rs.error_code
                            != "0"
                        ):

                            raise RuntimeError(
                                f"{rs.error_code}: "
                                f"{rs.error_msg}"
                            )

                        rows_for_stock = []

                        while rs.next():

                            rows_for_stock.append(
                                rs.get_row_data()
                            )

                        query_ok = True

                        break

                    except Exception as exc:

                        logger.warning(
                            f"[{symbol}] "
                            f"查询失败 "
                            f"({attempt}/"
                            f"{max_retries})："
                            f"{exc}"
                        )

                        # 断开当前连接
                        try:
                            bs.logout()
                        except Exception:
                            pass

                        if attempt < max_retries:

                            # 2 秒、4 秒
                            wait = 2 ** attempt

                            logger.info(
                                f"[{symbol}] "
                                f"{wait} 秒后重新连接..."
                            )

                            time.sleep(wait)

                            if not login_baostock():

                                logger.warning(
                                    f"[{symbol}] "
                                    f"baostock 重连失败，"
                                    f"稍后继续尝试"
                                )

                                time.sleep(3)

                        else:

                            logger.warning(
                                f"[{symbol}] "
                                f"{max_retries} 次查询均失败，"
                                f"跳过该股票"
                            )

                # ------------------------------------------------
                # 当前股票查询失败
                # ------------------------------------------------

                if not query_ok:

                    failed += 1
                    continue

                # ------------------------------------------------
                # 当前股票没有数据
                # ------------------------------------------------

                if not rows_for_stock:

                    empty += 1
                    continue

                # ------------------------------------------------
                # 添加到总数据
                # ------------------------------------------------

                for row in rows_for_stock:

                    all_rows.append(
                        [symbol] + row
                    )

                success += 1

                # ------------------------------------------------
                # 控制请求速度
                # ------------------------------------------------

                time.sleep(
                    request_interval
                )

                # ------------------------------------------------
                # 输出进度
                # ------------------------------------------------

                if (
                    index % 100 == 0
                    or index == len(tasks)
                ):

                    logger.info(
                        f"行情同步进度："
                        f"{index}/"
                        f"{len(tasks)} | "
                        f"成功 {success} | "
                        f"无数据 {empty} | "
                        f"失败 {failed}"
                    )

        finally:

            # ----------------------------------------------------
            # 最终退出 BaoStock
            # ----------------------------------------------------

            try:
                bs.logout()
            except Exception:
                pass

        # ========================================================
        # 没有获取到任何数据
        # ========================================================

        if not all_rows:

            logger.warning(
                "本次没有获取到任何新行情数据"
            )

            logger.info(
                f"行情同步结束："
                f"成功 {success} | "
                f"无数据 {empty} | "
                f"失败 {failed}"
            )

            return 0

        # ========================================================
        # 转换 DataFrame
        # ========================================================

        df = pd.DataFrame(
            all_rows,
            columns=[
                "symbol",
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "turnover",
            ],
        )

        # --------------------------------------------------------
        # 转换数值字段
        # --------------------------------------------------------

        for col in [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover",
        ]:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

        # --------------------------------------------------------
        # 删除无收盘价数据
        # --------------------------------------------------------

        df = df.dropna(
            subset=["close"]
        )

        # --------------------------------------------------------
        # 删除成交量为 0 的数据
        # --------------------------------------------------------

        df = df[
            df["volume"] > 0
        ]

        if df.empty:

            logger.warning(
                "过滤后没有有效行情数据"
            )

            return 0

        # ========================================================
        # 写入 SQLite
        # ========================================================

        count = len(df)

        try:

            with sqlite3.connect(
                self.db_path
            ) as conn:

                # ------------------------------------------------
                # 重要：
                #
                # 不再 DELETE 整个交易日。
                #
                # 使用 INSERT OR REPLACE，
                # 只更新实际成功获取到的股票。
                #
                # 这样即使部分股票获取失败，
                # 也不会把数据库中其他股票已有的数据删除。
                # ------------------------------------------------

                records = df[
                    [
                        "symbol",
                        "date",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "turnover",
                    ]
                ].itertuples(
                    index=False,
                    name=None,
                )

                conn.executemany(
                    """
                    INSERT OR REPLACE INTO stock_daily
                    (
                        symbol,
                        date,
                        open,
                        high,
                        low,
                        close,
                        volume,
                        turnover
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    records,
                )

                conn.commit()

        except Exception as exc:

            logger.error(
                f"SQLite 写入失败：{exc}"
            )

            return 0

        # ========================================================
        # 输出结果
        # ========================================================

        logger.info(
            f"sync_today_bulk："
            f"写入/更新 {count} 条数据"
        )

        logger.info(
            f"行情同步完成："
            f"成功 {success} | "
            f"无数据 {empty} | "
            f"失败 {failed}"
        )

        return count

    # ============================================================
    # 历史数据回填
    # ============================================================

    def backfill(
        self,
        symbols: list[str],
    ) -> None:
        """通过 baostock 批量回填历史日 K 线数据。

        数据采用后复权。

        容错机制：

        - 单只股票失败自动重试 3 次
        - 重试间隔递增
        - 每 200 只股票自动重连
        - 已入库股票自动跳过
        - 中断后可以重新运行继续
        """

        import time
        from datetime import date, timedelta

        import baostock as bs

        today_str = date.today().strftime(
            "%Y-%m-%d"
        )

        max_retries = 3

        reconnect_interval = 200

        # --------------------------------------------------------
        # 登录函数
        # --------------------------------------------------------

        def _login() -> bool:

            try:

                lg = bs.login()

                if lg.error_code != "0":

                    logger.error(
                        f"baostock 登录失败："
                        f"{lg.error_msg}"
                    )

                    return False

                return True

            except Exception as exc:

                logger.error(
                    f"baostock 登录异常："
                    f"{exc}"
                )

                return False

        # --------------------------------------------------------
        # 第一次登录
        # --------------------------------------------------------

        if not _login():

            return

        success = 0
        skipped = 0
        failed = 0

        since_reconnect = 0

        try:

            for i, symbol in enumerate(
                symbols
            ):

                last_date = (
                    self._get_last_date(
                        symbol
                    )
                )

                # ------------------------------------------------
                # 已经是最新
                # ------------------------------------------------

                if (
                    last_date
                    and last_date >= today_str
                ):

                    skipped += 1

                    if (
                        (i + 1) % 500 == 0
                    ):

                        logger.info(
                            f"已处理 "
                            f"{i + 1}/"
                            f"{len(symbols)}，"
                            f"成功 {success} "
                            f"跳过 {skipped} "
                            f"失败 {failed}"
                        )

                    continue

                # ------------------------------------------------
                # 定期重连
                # ------------------------------------------------

                since_reconnect += 1

                if (
                    since_reconnect
                    >= reconnect_interval
                ):

                    try:
                        bs.logout()
                    except Exception:
                        pass

                    time.sleep(1)

                    if not _login():

                        logger.error(
                            "重连失败，"
                            "终止回填"
                        )

                        return

                    since_reconnect = 0

                # ------------------------------------------------
                # 计算开始日期
                # ------------------------------------------------

                start = (
                    last_date
                    or self.start_date
                )

                if last_date:

                    start = (
                        date.fromisoformat(
                            last_date
                        )
                        + timedelta(days=1)
                    ).strftime(
                        "%Y-%m-%d"
                    )

                bs_code = (
                    self._to_baostock_code(
                        symbol
                    )
                )

                # ------------------------------------------------
                # 带重试查询
                # ------------------------------------------------

                rows = []
                query_ok = False

                for attempt in range(
                    max_retries
                ):

                    try:

                        rs = (
                            bs.query_history_k_data_plus(
                                bs_code,
                                (
                                    "date,open,high,low,"
                                    "close,volume,amount"
                                ),
                                start_date=start,
                                end_date=today_str,
                                frequency="d",
                                adjustflag="1",
                            )
                        )

                        if (
                            rs.error_code
                            != "0"
                        ):

                            raise RuntimeError(
                                rs.error_msg
                            )

                        rows = []

                        while rs.next():

                            rows.append(
                                rs.get_row_data()
                            )

                        query_ok = True

                        break

                    except Exception as exc:

                        if (
                            attempt
                            < max_retries - 1
                        ):

                            wait = (
                                2 ** (
                                    attempt + 1
                                )
                            )

                            logger.warning(
                                f"[{symbol}] "
                                f"第{attempt + 1}"
                                f"次失败："
                                f"{exc}，"
                                f"{wait}s 后重试"
                            )

                            time.sleep(
                                wait
                            )

                            try:
                                bs.logout()
                            except Exception:
                                pass

                            time.sleep(1)

                            _login()

                        else:

                            logger.warning(
                                f"[{symbol}] "
                                f"{max_retries}"
                                f"次重试均失败，"
                                f"跳过"
                            )

                # ------------------------------------------------
                # 查询失败
                # ------------------------------------------------

                if not query_ok:

                    failed += 1
                    continue

                # ------------------------------------------------
                # 没有数据
                # ------------------------------------------------

                if not rows:

                    skipped += 1
                    continue

                # ------------------------------------------------
                # DataFrame
                # ------------------------------------------------

                df = pd.DataFrame(
                    rows,
                    columns=rs.fields,
                )

                for col in [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "amount",
                ]:

                    df[col] = pd.to_numeric(
                        df[col],
                        errors="coerce",
                    )

                df = df.dropna(
                    subset=["close"]
                )

                df = df[
                    df["volume"] > 0
                ]

                if df.empty:

                    skipped += 1
                    continue

                # ------------------------------------------------
                # 整理字段
                # ------------------------------------------------

                df["symbol"] = symbol

                df = df.rename(
                    columns={
                        "amount": "turnover"
                    }
                )

                df = df[
                    [
                        "symbol",
                        "date",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "turnover",
                    ]
                ]

                # ------------------------------------------------
                # 写入 SQLite
                # ------------------------------------------------

                try:

                    with sqlite3.connect(
                        self.db_path
                    ) as conn:

                        records = df[
                            [
                                "symbol",
                                "date",
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume",
                                "turnover",
                            ]
                        ].itertuples(
                            index=False,
                            name=None,
                        )

                        conn.executemany(
                            """
                            INSERT OR REPLACE
                            INTO stock_daily
                            (
                                symbol,
                                date,
                                open,
                                high,
                                low,
                                close,
                                volume,
                                turnover
                            )
                            VALUES
                            (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            records,
                        )

                        conn.commit()

                except Exception as exc:

                    logger.warning(
                        f"[{symbol}] "
                        f"数据库写入失败："
                        f"{exc}"
                    )

                    failed += 1
                    continue

                success += 1

                # ------------------------------------------------
                # 输出进度
                # ------------------------------------------------

                if (
                    (i + 1) % 500 == 0
                ):

                    logger.info(
                        f"已处理 "
                        f"{i + 1}/"
                        f"{len(symbols)}，"
                        f"成功 {success} "
                        f"跳过 {skipped} "
                        f"失败 {failed}"
                    )

        finally:

            try:
                bs.logout()
            except Exception:
                pass

        logger.info(
            f"回填完成 — "
            f"成功: {success} | "
            f"跳过: {skipped} | "
            f"失败: {failed}"
        )

    # ============================================================
    # 获取全市场股票列表
    # ============================================================

    def get_all_symbols(
        self,
    ) -> list[str]:
        """通过 baostock 获取全市场 A 股代码列表。"""

        import baostock as bs

        try:

            lg = bs.login()

            if lg.error_code != "0":

                logger.error(
                    f"baostock 登录失败："
                    f"{lg.error_msg}"
                )

                return []

            rs = bs.query_stock_basic(
                code_name="",
                code="",
            )

            symbols = []

            while rs.next():

                row = rs.get_row_data()

                code = row[0]

                status = row[4]

                stock_type = row[5]

                # status=1：上市
                # stock_type=1：股票

                if (
                    status == "1"
                    and stock_type == "1"
                ):

                    symbols.append(
                        code.split(".")[1]
                    )

            logger.info(
                f"获取股票列表完成，"
                f"共 {len(symbols)} 只"
            )

            return symbols

        except Exception as exc:

            logger.error(
                f"获取股票列表失败："
                f"{exc}"
            )

            return []

        finally:

            try:
                bs.logout()
            except Exception:
                pass

    # ============================================================
    # 获取本地股票列表
    # ============================================================

    def get_local_symbols(
        self,
    ) -> list[str]:
        """获取本地数据库中的股票代码列表。"""

        with sqlite3.connect(
            self.db_path
        ) as conn:

            rows = conn.execute(
                """
                SELECT DISTINCT symbol
                FROM stock_daily
                """
            ).fetchall()

        return [
            row[0]
            for row in rows
        ]
