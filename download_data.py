"""批量下载 A 股日线数据到本地。

使用方法:
    python download_data.py --start 2020-01-01 --end 2024-12-31 --limit 100
    python download_data.py --symbols 000001.SZ,600519.SH,300750.SZ
    python download_data.py --index-components hs300 --random 20
    python download_data.py --source auto --workers 2 --retries 3
"""

import sys
sys.path.insert(0, '/Users/aqichita/projects/stock-backtest')

import argparse
import random
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

from src.data.manager import DataManager
from src.data.storage import DataStorage


DEFAULT_SYMBOLS = [
    "000001.SZ",  # 平安银行
    "600036.SH",  # 招商银行
    "601398.SH",  # 工商银行
    "600519.SH",  # 贵州茅台
    "000858.SZ",  # 五粮液
    "300750.SZ",  # 宁德时代
    "002594.SZ",  # 比亚迪
    "601012.SH",  # 隆基绿能
    "600276.SH",  # 恒瑞医药
    "000538.SZ",  # 云南白药
    "000333.SZ",  # 美的集团
    "600887.SH",  # 伊利股份
]

SUPPORTED_SOURCES = ("auto", "baostock", "akshare", "sina")
RAW_DATA_DIR = Path("data/raw")
RAW_SQLITE_PATH = RAW_DATA_DIR / "stock_data.db"


def normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def is_a_share_symbol(symbol: str) -> bool:
    symbol = normalize_symbol(symbol)
    if "." not in symbol:
        return False
    code, exchange = symbol.split(".", 1)
    return exchange in {"SH", "SZ", "BJ"} and code.isdigit() and len(code) == 6


def download_single_stock(
    symbol: str,
    start_date: str,
    end_date: str,
    storage: DataStorage,
    force_update: bool = False,
    source: Literal["auto", "baostock", "akshare", "sina"] = "auto",
    retries: int = 3,
    retry_delay: float = 1.5,
) -> dict:
    """下载单只股票数据并保存到本地。"""
    symbol = normalize_symbol(symbol)

    if not is_a_share_symbol(symbol):
        return {
            "symbol": symbol,
            "status": "invalid",
            "rows": 0,
            "message": "仅支持 A 股/BJ 股票代码，格式如 000001.SZ",
        }

    if not force_update:
        existing = storage.load_daily_data(symbol, start=start_date, end=end_date)
        if existing is not None and len(existing) > 0:
            return {
                "symbol": symbol,
                "status": "skipped",
                "rows": len(existing),
                "message": "数据已存在，跳过下载",
            }

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            dm = DataManager()
            df = dm.get_daily_data(symbol, start=start_date, end=end_date, source=source)
            if df is None or df.empty:
                raise ValueError("无数据返回")

            storage.save_daily_data(symbol, df)
            return {
                "symbol": symbol,
                "status": "success",
                "rows": len(df),
                "message": f"成功下载 {len(df)} 条数据 (source={source}, attempt={attempt})",
            }
        except Exception as e:
            last_error = str(e)
            if attempt < retries:
                time.sleep(retry_delay * attempt)

    return {
        "symbol": symbol,
        "status": "error",
        "rows": 0,
        "message": f"重试 {retries} 次后仍失败: {last_error}",
    }


def create_storage(preferred_backend: str = "auto") -> tuple[DataStorage, str]:
    """创建存储后端，优先 parquet，不可用时回退 sqlite。"""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if preferred_backend in {"auto", "parquet"}:
        try:
            import pyarrow  # noqa: F401
            return DataStorage(backend="parquet", data_dir=str(RAW_DATA_DIR)), "parquet"
        except ImportError:
            if preferred_backend == "parquet":
                raise

    return DataStorage(backend="sqlite", db_path=str(RAW_SQLITE_PATH)), "sqlite"


def download_all(
    symbols: list[str],
    start_date: str,
    end_date: str,
    max_workers: int = 4,
    force_update: bool = False,
    source: Literal["auto", "baostock", "akshare", "sina"] = "auto",
    retries: int = 3,
    retry_delay: float = 1.5,
    storage_backend: str = "auto",
):
    """批量下载多只股票数据。"""
    storage, actual_backend = create_storage(storage_backend)

    print(f"开始下载 {len(symbols)} 只股票数据...")
    print(f"时间范围: {start_date} ~ {end_date}")
    print(f"数据源模式: {source}")
    print(f"输出目录: {RAW_DATA_DIR}")
    print(f"存储后端: {actual_backend}")
    print(f"并发数: {max_workers}")
    print(f"重试次数: {retries}")
    print("-" * 60)

    results = []
    success_count = 0
    failed_count = 0
    skipped_count = 0
    invalid_count = 0
    total_rows = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(
                download_single_stock,
                symbol,
                start_date,
                end_date,
                storage,
                force_update,
                source,
                retries,
                retry_delay,
            ): symbol
            for symbol in symbols
        }

        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                results.append(result)

                if result["status"] == "success":
                    success_count += 1
                    total_rows += result["rows"]
                    print(f"✅ {symbol}: {result['message']}")
                elif result["status"] == "skipped":
                    skipped_count += 1
                    total_rows += result["rows"]
                    print(f"⏭️  {symbol}: {result['message']}")
                elif result["status"] == "invalid":
                    invalid_count += 1
                    print(f"⚠️  {symbol}: {result['message']}")
                else:
                    failed_count += 1
                    print(f"❌ {symbol}: {result['message']}")
            except Exception as e:
                failed_count += 1
                print(f"❌ {symbol}: 异常 - {e}")

    print("-" * 60)
    print("下载完成!")
    print(f"  成功: {success_count}")
    print(f"  跳过: {skipped_count}")
    print(f"  无效代码: {invalid_count}")
    print(f"  失败: {failed_count}")
    print(f"  总数据行数: {total_rows}")

    return results


def get_index_components(index_code: str) -> list[str]:
    """获取指数成分股列表。"""
    try:
        import akshare as ak
        df = ak.index_stock_cons_weight_csindex(symbol=index_code)
        symbols = []
        for _, row in df.iterrows():
            code = str(row["成分券代码"]).zfill(6)
            if code.startswith(("6", "9")):
                symbols.append(f"{code}.SH")
            elif code.startswith(("4", "8")):
                symbols.append(f"{code}.BJ")
            else:
                symbols.append(f"{code}.SZ")
        return symbols
    except Exception as e:
        print(f"获取指数 {index_code} 成分股失败: {e}")
        return []


def get_hs300_components() -> list[str]:
    return get_index_components("000300")


def get_zz500_components() -> list[str]:
    return get_index_components("000905")


def get_stock_list_a() -> list[str]:
    """获取全部 A 股列表。"""
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        symbols = []
        for _, row in df.iterrows():
            code = str(row["代码"]).zfill(6)
            if code.startswith(("6", "9")):
                symbols.append(f"{code}.SH")
            elif code.startswith(("4", "8")):
                symbols.append(f"{code}.BJ")
            else:
                symbols.append(f"{code}.SZ")
        return symbols
    except Exception as e:
        print(f"获取 A 股列表失败: {e}")
        return []


def resolve_symbols(args) -> list[str]:
    if args.symbols:
        symbols = [normalize_symbol(s) for s in args.symbols.split(",") if s.strip()]
    elif args.index_components == "hs300":
        print("正在获取沪深300成分股列表...")
        symbols = get_hs300_components()
    elif args.index_components == "zz500":
        print("正在获取中证500成分股列表...")
        symbols = get_zz500_components()
    elif args.index_components == "a":
        print("正在获取全部A股列表...")
        symbols = get_stock_list_a()
    else:
        symbols = DEFAULT_SYMBOLS.copy()

    symbols = list(dict.fromkeys(symbols))

    if args.random:
        sample_size = min(args.random, len(symbols))
        symbols = random.sample(symbols, sample_size)
        print(f"已随机选择 {sample_size} 只股票")

    if args.limit and len(symbols) > args.limit:
        symbols = symbols[:args.limit]
        print(f"已限制下载前 {args.limit} 只股票")

    return symbols


def main():
    parser = argparse.ArgumentParser(description="批量下载 A 股股票数据")
    parser.add_argument("--start", type=str, default="2020-01-01", help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=datetime.now().strftime("%Y-%m-%d"), help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--symbols", type=str, help="股票代码列表，逗号分隔，如: 000001.SZ,600519.SH")
    parser.add_argument("--limit", type=int, default=None, help="限制下载股票数量")
    parser.add_argument("--random", type=int, default=None, help="从候选股票中随机抽取 N 只")
    parser.add_argument("--workers", type=int, default=2, help="并发下载数 (默认2)")
    parser.add_argument("--force", action="store_true", help="强制更新，即使数据已存在也重新下载")
    parser.add_argument("--retries", type=int, default=3, help="单只股票失败后的重试次数")
    parser.add_argument("--retry-delay", type=float, default=1.5, help="首次重试等待秒数，后续按 attempt 倍数递增")
    parser.add_argument("--source", type=str, default="auto", choices=SUPPORTED_SOURCES, help="数据源模式，默认 auto(优先 Baostock)")
    parser.add_argument("--index-components", type=str, choices=["hs300", "zz500", "a"], help="下载指数成分股 (hs300=沪深300, zz500=中证500, a=全部A股)")
    parser.add_argument("--storage-backend", type=str, default="auto", choices=["auto", "parquet", "sqlite"], help="存储后端，默认 auto(parquet 不可用时回退 sqlite)")

    args = parser.parse_args()
    symbols = resolve_symbols(args)
    if not symbols:
        raise SystemExit("没有可下载的股票代码，请检查参数或上游列表接口")

    download_all(
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        max_workers=args.workers,
        force_update=args.force,
        source=args.source,
        retries=args.retries,
        retry_delay=args.retry_delay,
        storage_backend=args.storage_backend,
    )


if __name__ == "__main__":
    main()
