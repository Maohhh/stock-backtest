"""批量下载期货全品种「加权(指数)」15 分钟 K 线数据到本地。

数据源: 东方财富历史 K 线接口 (push2his.eastmoney.com)
    - 加权/指数序列位于市场号 159，代码形如 159.rbfi(螺纹钢加权)、159.cufi(沪铜加权)。
    - klt=15 表示 15 分钟周期；fqt=1 表示前复权。

品种清单与加权代码映射见 src/data/futures_weighted_symbols.json，
可用 --refresh-symbols 通过东财搜索接口重新解析。

使用方法:
    # 默认下载全品种近 5 年 15 分钟加权数据
    python download_futures_15m.py

    # 指定时间范围与并发
    python download_futures_15m.py --start 2020-01-01 --end 2025-05-30 --workers 4

    # 只下载部分品种(按中文名或加权代码)
    python download_futures_15m.py --products 螺纹钢,沪铜,rbfi

    # 强制覆盖已存在文件
    python download_futures_15m.py --force

    # 重新解析品种->加权代码映射(需要东财搜索接口可用)
    python download_futures_15m.py --refresh-symbols

注意: 该接口主机 push2his.eastmoney.com 在部分受限网络/云环境中会返回 503
      (upstream connect error)。脚本启动时会做连通性预检，不通则给出提示并退出。
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
SYMBOLS_PATH = ROOT / "src" / "data" / "futures_weighted_symbols.json"
OUTPUT_DIR = ROOT / "data" / "futures_15m"

KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
SEARCH_URL = "https://searchapi.eastmoney.com/api/suggest/get"
SEARCH_TOKEN = "D43BF722C8E33BDC906FB84D85E326E8"

# klines 字段: 时间, 开, 收, 高, 低, 成交量, 成交额, 振幅
KLINE_FIELDS2 = "f51,f52,f53,f54,f55,f56,f57,f58"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://quote.eastmoney.com/",
}


def load_symbols() -> dict:
    """加载 品种名 -> {secid, code, emname, jys} 映射。"""
    if not SYMBOLS_PATH.exists():
        raise SystemExit(f"未找到品种映射文件: {SYMBOLS_PATH}，请先运行 --refresh-symbols")
    with open(SYMBOLS_PATH, encoding="utf-8") as f:
        return json.load(f)


def refresh_symbols() -> dict:
    """通过东财搜索接口重新解析期货全品种的加权(159 市场)代码。"""
    session = requests.Session()
    session.headers.update(HEADERS)

    def search(query: str) -> list:
        for attempt in range(4):
            try:
                r = session.get(
                    SEARCH_URL,
                    params={"input": query, "type": "14", "token": SEARCH_TOKEN, "count": "12"},
                    timeout=12,
                )
                return r.json()["QuotationCodeTable"]["Data"] or []
            except Exception:
                time.sleep(1.0)
        return []

    # 标准名 -> 候选搜索词(东财命名与交易所简称不完全一致)
    products = {
        "沪铜": ["沪铜加权"], "沪铝": ["沪铝加权"], "沪锌": ["沪锌加权"], "沪铅": ["沪铅加权"],
        "沪镍": ["沪镍加权"], "沪锡": ["沪锡加权"], "黄金": ["沪金加权"], "白银": ["沪银加权"],
        "螺纹钢": ["螺纹钢加权"], "线材": ["线材加权"], "热卷": ["热卷加权"], "不锈钢": ["不锈钢加权"],
        "燃油": ["燃油加权"], "沥青": ["沥青加权"], "橡胶": ["橡胶加权"], "纸浆": ["纸浆加权"],
        "氧化铝": ["氧化铝加权"], "丁二烯橡胶": ["合成橡胶加权", "丁二烯橡胶加权"],
        "原油": ["原油加权"], "低硫燃油": ["低硫燃油加权"], "20号胶": ["20号胶加权"],
        "国际铜": ["国际铜加权"], "集运欧线": ["集运欧线加权", "欧线集运加权"],
        "豆一": ["豆一加权"], "豆二": ["豆二加权"], "豆粕": ["豆粕加权"], "豆油": ["豆油加权"],
        "棕榈油": ["棕榈油加权"], "玉米": ["玉米加权"], "玉米淀粉": ["淀粉加权"], "鸡蛋": ["鸡蛋加权"],
        "生猪": ["生猪加权"], "焦炭": ["焦炭加权"], "焦煤": ["焦煤加权"], "铁矿石": ["铁矿石加权"],
        "聚乙烯": ["塑料加权"], "聚丙烯": ["聚丙烯加权"], "聚氯乙烯": ["PVC加权"],
        "乙二醇": ["乙二醇加权"], "苯乙烯": ["苯乙烯加权"], "液化石油气": ["LPG加权", "液化气加权"],
        "纤维板": ["纤维板加权"], "胶合板": ["胶合板加权"],
        "白糖": ["白糖加权"], "棉花": ["棉花加权"], "棉纱": ["棉纱加权"], "PTA": ["PTA加权"],
        "甲醇": ["甲醇加权"], "玻璃": ["玻璃加权"], "纯碱": ["纯碱加权"], "菜粕": ["菜粕加权"],
        "菜油": ["菜油加权"], "花生": ["花生加权"], "苹果": ["苹果加权"], "红枣": ["红枣加权"],
        "短纤": ["短纤加权"], "尿素": ["尿素加权"], "硅铁": ["硅铁加权"], "锰硅": ["锰硅加权"],
        "烧碱": ["烧碱加权"], "对二甲苯": ["对二甲苯加权"], "强麦": ["强麦加权"],
        "油菜籽": ["菜籽加权", "油菜籽加权"], "瓶片": ["瓶片加权"], "普麦": ["普麦加权"],
        "粳稻": ["粳稻加权"], "早籼稻": ["早籼稻加权"],
        "工业硅": ["工业硅加权"], "碳酸锂": ["碳酸锂加权"], "多晶硅": ["多晶硅加权"],
    }

    mapping, missing = {}, []
    for name, queries in products.items():
        pick = None
        for q in queries:
            for it in search(q):
                if it.get("MktNum") == "159" and "加权" in it.get("Name", ""):
                    pick = it
                    break
            if pick:
                break
        if pick:
            mapping[name] = {
                "secid": pick["QuoteID"],
                "code": pick["Code"],
                "emname": pick["Name"],
                "jys": pick.get("JYS"),
            }
        else:
            missing.append(name)
        time.sleep(0.2)

    SYMBOLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SYMBOLS_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"已解析 {len(mapping)} 个品种 -> {SYMBOLS_PATH}")
    if missing:
        print(f"未解析(可能无加权指数): {missing}")
    return mapping


def preflight(session: requests.Session) -> bool:
    """连通性预检: 历史 K 线主机是否可达。"""
    try:
        r = session.get(
            KLINE_URL,
            params={
                "secid": "159.rbfi", "klt": "15", "fqt": "1",
                "fields1": "f1,f2", "fields2": KLINE_FIELDS2,
                "lmt": "5", "end": "20500101",
            },
            timeout=15,
        )
        return r.status_code == 200 and r.text.strip().startswith("{")
    except Exception:
        return False


def fetch_kline(
    session: requests.Session,
    secid: str,
    start: str,
    end: str,
    retries: int = 4,
    retry_delay: float = 2.0,
):
    """拉取单个加权品种的 15 分钟 K 线，返回 (klines_list, name) 或 (None, None)。"""
    params = {
        "secid": secid,
        "klt": "15",
        "fqt": "1",
        "fields1": "f1,f2,f3,f4,f5",
        "fields2": KLINE_FIELDS2,
        "beg": start.replace("-", ""),
        "end": end.replace("-", ""),
        "lmt": "1000000",
    }
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = session.get(KLINE_URL, params=params, timeout=25)
            if r.status_code == 200 and r.text.strip().startswith("{"):
                data = (r.json() or {}).get("data") or {}
                return data.get("klines") or [], data.get("name")
            last_err = f"HTTP {r.status_code}: {r.text[:60]}"
        except Exception as e:
            last_err = str(e)
        if attempt < retries:
            time.sleep(retry_delay * attempt)
    raise RuntimeError(last_err or "未知错误")


def klines_to_df(klines: list):
    """将东财 klines 字符串列表转为标准化 DataFrame。"""
    import pandas as pd

    rows = []
    for line in klines:
        parts = line.split(",")
        if len(parts) < 7:
            continue
        rows.append({
            "datetime": parts[0],
            "open": float(parts[1]),
            "close": float(parts[2]),
            "high": float(parts[3]),
            "low": float(parts[4]),
            "volume": float(parts[5]),
            "amount": float(parts[6]),
        })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df = df[["datetime", "open", "high", "low", "close", "volume", "amount"]]
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def save_df(df, code: str, backend: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if backend == "parquet":
        path = OUTPUT_DIR / f"{code}.parquet"
        df.to_parquet(path, index=False)
    else:
        path = OUTPUT_DIR / f"{code}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def pick_backend(preferred: str) -> str:
    if preferred in {"auto", "parquet"}:
        try:
            import pyarrow  # noqa: F401
            return "parquet"
        except ImportError:
            if preferred == "parquet":
                raise SystemExit("未安装 pyarrow，无法使用 parquet 后端，请 pip install pyarrow")
    return "csv"


def download_one(name, info, start, end, backend, force, retries, retry_delay):
    code = info["code"]
    session = requests.Session()
    session.headers.update(HEADERS)

    out = OUTPUT_DIR / f"{code}.{ 'parquet' if backend=='parquet' else 'csv'}"
    if out.exists() and not force:
        return {"name": name, "status": "skipped", "rows": 0, "message": "文件已存在，跳过"}

    try:
        klines, em_name = fetch_kline(session, info["secid"], start, end, retries, retry_delay)
        if not klines:
            return {"name": name, "status": "empty", "rows": 0, "message": "无数据返回"}
        df = klines_to_df(klines)
        if df is None or df.empty:
            return {"name": name, "status": "empty", "rows": 0, "message": "解析后无有效数据"}
        path = save_df(df, code, backend)
        span = f"{df['datetime'].iloc[0]:%Y-%m-%d} ~ {df['datetime'].iloc[-1]:%Y-%m-%d}"
        return {"name": name, "status": "success", "rows": len(df),
                "message": f"{len(df)} 行 [{span}] -> {path.name}"}
    except Exception as e:
        return {"name": name, "status": "error", "rows": 0, "message": str(e)}


def resolve_products(symbols: dict, products_arg: str) -> dict:
    if not products_arg:
        return symbols
    wanted = {p.strip() for p in products_arg.split(",") if p.strip()}
    selected = {}
    for name, info in symbols.items():
        if name in wanted or info["code"] in wanted or info["secid"] in wanted:
            selected[name] = info
    missing = wanted - {n for n in selected} - {selected[n]["code"] for n in selected}
    if not selected:
        raise SystemExit(f"未匹配到任何品种: {products_arg}")
    return selected


def main():
    parser = argparse.ArgumentParser(description="下载期货全品种 15 分钟加权数据")
    default_end = datetime.now().strftime("%Y-%m-%d")
    default_start = (datetime.now() - timedelta(days=365 * 5 + 2)).strftime("%Y-%m-%d")
    parser.add_argument("--start", default=default_start, help="开始日期 YYYY-MM-DD(默认近5年)")
    parser.add_argument("--end", default=default_end, help="结束日期 YYYY-MM-DD(默认今天)")
    parser.add_argument("--products", default=None, help="只下载指定品种，逗号分隔(中文名或加权代码)")
    parser.add_argument("--workers", type=int, default=3, help="并发数(默认3)")
    parser.add_argument("--force", action="store_true", help="覆盖已存在文件")
    parser.add_argument("--retries", type=int, default=4, help="单品种失败重试次数")
    parser.add_argument("--retry-delay", type=float, default=2.0, help="重试基础等待秒数")
    parser.add_argument("--backend", default="auto", choices=["auto", "parquet", "csv"], help="存储后端")
    parser.add_argument("--refresh-symbols", action="store_true", help="重新解析品种->加权代码映射后退出")
    parser.add_argument("--skip-preflight", action="store_true", help="跳过连通性预检")
    args = parser.parse_args()

    if args.refresh_symbols:
        refresh_symbols()
        return

    symbols = load_symbols()
    targets = resolve_products(symbols, args.products)
    backend = pick_backend(args.backend)

    session = requests.Session()
    session.headers.update(HEADERS)
    if not args.skip_preflight:
        print("连通性预检 push2his.eastmoney.com ...")
        if not preflight(session):
            print("❌ 历史 K 线主机不可达 (push2his.eastmoney.com 返回 503/连接被拒)。")
            print("   该主机在受限网络/云环境常被风控或网络策略拦截。请在可访问该主机的")
            print("   网络环境(本地机器或放开出网策略的环境)中重试，或加 --skip-preflight 强行尝试。")
            raise SystemExit(2)
        print("✅ 预检通过\n")

    print(f"开始下载 {len(targets)} 个品种 15 分钟加权数据")
    print(f"时间范围: {args.start} ~ {args.end}")
    print(f"输出目录: {OUTPUT_DIR}  存储后端: {backend}  并发: {args.workers}")
    print("-" * 64)

    results, ok, skip, empty, err, total_rows = [], 0, 0, 0, 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(download_one, name, info, args.start, args.end,
                      backend, args.force, args.retries, args.retry_delay): name
            for name, info in targets.items()
        }
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            icon = {"success": "✅", "skipped": "⏭️ ", "empty": "⚪", "error": "❌"}.get(res["status"], "?")
            print(f"{icon} {res['name']}: {res['message']}")
            if res["status"] == "success":
                ok += 1; total_rows += res["rows"]
            elif res["status"] == "skipped":
                skip += 1
            elif res["status"] == "empty":
                empty += 1
            else:
                err += 1

    print("-" * 64)
    print(f"完成! 成功 {ok} / 跳过 {skip} / 空 {empty} / 失败 {err}  总行数 {total_rows}")
    if err:
        sys.exit(1)


if __name__ == "__main__":
    main()
