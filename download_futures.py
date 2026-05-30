"""下载国内期货全品种 15 分钟「加权」连续数据到本地仓库，用于回测。

什么是「加权」(weighted continuous)?
    单个期货品种(如螺纹钢 RB)同时有多个在交易的月份合约(RB2510、RB2601 ...)。
    「加权」连续序列是把同一品种所有在交易合约，按各自的**持仓量(open interest)**
    在每一根 K 线上加权合成的一条价格序列:

        close_weighted(t) = Σ_i close_i(t) * oi_i(t) / Σ_i oi_i(t)

    它不像「主力连续」那样在换月时产生跳空，更适合做连续回测/指标计算。

数据源与限制:
    新浪财经 getFewMinLine 免费接口。该接口对每个合约**最多只返回最近 1023 根**
    15 分钟 K 线(约 2 个月)，因此加权序列的历史深度同样约 ~2 个月，这是免费源的硬限制。
    接口字段: d=时间 o/h/l/c=OHLC v=成交量 p=持仓量。

使用方法:
    python download_futures.py                      # 全品种 15min 加权
    python download_futures.py --products RB,M,CU    # 指定品种
    python download_futures.py --period 5            # 改周期(1/5/15/30/60)
    python download_futures.py --workers 8           # 并发抓取合约
"""

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests


class RateLimiter:
    """线程安全的全局最小请求间隔，避免触发新浪限流。"""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            sleep_for = self._next - now
            self._next = max(now, self._next) + self.min_interval
        if sleep_for > 0:
            time.sleep(sleep_for)

# ---------------------------------------------------------------------------
# 全品种定义: 品种码 -> (交易所, 中文名)
# ---------------------------------------------------------------------------
PRODUCTS: dict[str, tuple[str, str]] = {
    # 上期所 SHFE
    "CU": ("SHFE", "沪铜"), "AL": ("SHFE", "沪铝"), "ZN": ("SHFE", "沪锌"),
    "PB": ("SHFE", "沪铅"), "NI": ("SHFE", "沪镍"), "SN": ("SHFE", "沪锡"),
    "AU": ("SHFE", "沪金"), "AG": ("SHFE", "沪银"), "RB": ("SHFE", "螺纹钢"),
    "WR": ("SHFE", "线材"), "HC": ("SHFE", "热卷"), "SS": ("SHFE", "不锈钢"),
    "BU": ("SHFE", "沥青"), "RU": ("SHFE", "橡胶"), "FU": ("SHFE", "燃油"),
    "SP": ("SHFE", "纸浆"), "AO": ("SHFE", "氧化铝"), "BR": ("SHFE", "合成橡胶"),
    # 能源中心 INE
    "SC": ("INE", "原油"), "LU": ("INE", "低硫燃料油"), "NR": ("INE", "20号胶"),
    "BC": ("INE", "国际铜"), "EC": ("INE", "集运欧线"),
    # 大商所 DCE
    "A": ("DCE", "豆一"), "B": ("DCE", "豆二"), "M": ("DCE", "豆粕"),
    "Y": ("DCE", "豆油"), "P": ("DCE", "棕榈油"), "C": ("DCE", "玉米"),
    "CS": ("DCE", "玉米淀粉"), "JD": ("DCE", "鸡蛋"), "L": ("DCE", "塑料"),
    "V": ("DCE", "PVC"), "PP": ("DCE", "聚丙烯"), "J": ("DCE", "焦炭"),
    "JM": ("DCE", "焦煤"), "I": ("DCE", "铁矿石"), "EG": ("DCE", "乙二醇"),
    "EB": ("DCE", "苯乙烯"), "PG": ("DCE", "液化石油气"), "RR": ("DCE", "粳米"),
    "LH": ("DCE", "生猪"), "FB": ("DCE", "纤维板"), "BB": ("DCE", "胶合板"),
    "LG": ("DCE", "原木"),
    # 郑商所 CZCE
    "SR": ("CZCE", "白糖"), "CF": ("CZCE", "棉花"), "CY": ("CZCE", "棉纱"),
    "TA": ("CZCE", "PTA"), "MA": ("CZCE", "甲醇"), "FG": ("CZCE", "玻璃"),
    "RM": ("CZCE", "菜粕"), "OI": ("CZCE", "菜油"), "SF": ("CZCE", "硅铁"),
    "SM": ("CZCE", "锰硅"), "AP": ("CZCE", "苹果"), "CJ": ("CZCE", "红枣"),
    "UR": ("CZCE", "尿素"), "SA": ("CZCE", "纯碱"), "PF": ("CZCE", "短纤"),
    "PK": ("CZCE", "花生"), "SH": ("CZCE", "烧碱"), "PX": ("CZCE", "对二甲苯"),
    "WH": ("CZCE", "强麦"), "PM": ("CZCE", "普麦"), "RI": ("CZCE", "早籼稻"),
    "JR": ("CZCE", "粳稻"), "LR": ("CZCE", "晚籼稻"), "RS": ("CZCE", "菜籽"),
    # 中金所 CFFEX
    "IF": ("CFFEX", "沪深300"), "IC": ("CFFEX", "中证500"),
    "IH": ("CFFEX", "上证50"), "IM": ("CFFEX", "中证1000"),
    "TS": ("CFFEX", "2年国债"), "TF": ("CFFEX", "5年国债"),
    "T": ("CFFEX", "10年国债"), "TL": ("CFFEX", "30年国债"),
    # 广期所 GFEX
    "SI": ("GFEX", "工业硅"), "LC": ("GFEX", "碳酸锂"), "PS": ("GFEX", "多晶硅"),
}

BASE_URL = (
    "https://stock2.finance.sina.com.cn/futures/api/jsonp.php/cb/"
    "InnerFuturesNewService.getFewMinLine?symbol={symbol}&type={period}"
)


def candidate_months(back: int = 3, forward: int = 19) -> list[str]:
    """生成 YYMM 候选月份(4 位)，覆盖近月到远月在交易合约。"""
    now = datetime.now()
    months = []
    for delta in range(-back, forward + 1):
        y = now.year + (now.month - 1 + delta) // 12
        m = (now.month - 1 + delta) % 12 + 1
        months.append(f"{y % 100:02d}{m:02d}")
    return months


class Throttled(Exception):
    """新浪反爬封禁(HTTP 456)，需要冷却后重试。"""


def fetch_contract(session: requests.Session, contract: str, period: int,
                   limiter: "RateLimiter", max_attempts: int = 8) -> pd.DataFrame | None:
    """抓取单个合约的 N 分钟 K 线。

    用 HTTP 状态码区分三种情况:
      * 456(反爬封禁) 或网络异常 -> 限流，长退避后重试，直到成功
      * 403/404                 -> 该合约已下架/不可查，返回 None(不重试)
      * 200 但无数据数组         -> 合约不存在，返回 None(不重试)
      * 200 且有数组             -> 返回解析后的 DataFrame

    这样保证每个真实存在的合约最终一定被抓到(真加权完整性)，而不会把限流误判为空。
    """
    url = BASE_URL.format(symbol=contract, period=period)
    backoff = [5, 15, 30, 60, 120, 180, 240]
    for attempt in range(max_attempts):
        try:
            limiter.wait()
            resp = session.get(url, timeout=20)
            if resp.status_code == 456:
                raise Throttled(contract)
            if resp.status_code in (403, 404):
                return None  # 老合约已下架/不可查, 跳过
            resp.raise_for_status()
            text = resp.text
            match = re.search(r"\[.*\]", text, re.S)
            if not match:
                return None  # 合约不存在
            records = json.loads(match.group(0))
            if not records:
                return None
            df = pd.DataFrame(records)
            df = df.rename(columns={"d": "datetime", "o": "open", "h": "high",
                                    "l": "low", "c": "close", "v": "volume",
                                    "p": "open_interest"})
            df["datetime"] = pd.to_datetime(df["datetime"])
            for col in ["open", "high", "low", "close", "volume", "open_interest"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["close"])
            df["contract"] = contract
            return df
        except Throttled:
            if attempt < max_attempts - 1:
                time.sleep(backoff[min(attempt, len(backoff) - 1)])
        except Exception:
            if attempt < max_attempts - 1:
                time.sleep(3 * (attempt + 1))
    # 多次重试仍被限流：交还调用方判定为失败(而非空)
    raise Throttled(contract)


def build_weighted(frames: list[pd.DataFrame], min_dt: pd.Timestamp) -> pd.DataFrame:
    """把同品种多合约按持仓量加权合成连续序列。"""
    df = pd.concat(frames, ignore_index=True)
    # 只保留目标窗口内、且持仓量为正的样本
    df = df[(df["datetime"] >= min_dt) & (df["open_interest"] > 0)].copy()
    if df.empty:
        return pd.DataFrame()

    oi = df["open_interest"]
    for col in ["open", "high", "low", "close"]:
        df[f"_w_{col}"] = df[col] * oi

    grouped = df.groupby("datetime")
    out = pd.DataFrame({
        "open": grouped["_w_open"].sum() / grouped["open_interest"].sum(),
        "high": grouped["_w_high"].sum() / grouped["open_interest"].sum(),
        "low": grouped["_w_low"].sum() / grouped["open_interest"].sum(),
        "close": grouped["_w_close"].sum() / grouped["open_interest"].sum(),
        "volume": grouped["volume"].sum(),
        "open_interest": grouped["open_interest"].sum(),
        "n_contracts": grouped["contract"].nunique(),
    }).reset_index()
    out = out.sort_values("datetime").reset_index(drop=True)
    for col in ["open", "high", "low", "close"]:
        out[col] = out[col].round(4)
    return out


def download_product(product: str, period: int, months: list[str],
                     min_dt: pd.Timestamp, session: requests.Session,
                     limiter: "RateLimiter",
                     stop_after_empty: int = 4) -> tuple[pd.DataFrame, bool]:
    """下载并合成单个品种的加权序列。

    顺序抓取(对反爬更友好): 按月份升序探测合约，找到首个真实合约后，
    若连续 stop_after_empty 个月份不存在则停止探测远月。

    返回 (加权DataFrame, complete)。complete=False 表示有合约被限流抓取失败，
    数据不完整，应由上层冷却后重试，避免落地退化的加权序列。
    """
    frames: list[pd.DataFrame] = []
    complete = True
    seen_real = False
    consecutive_empty = 0
    for ym in months:
        contract = f"{product}{ym}"
        try:
            df = fetch_contract(session, contract, period, limiter)
        except Throttled:
            complete = False  # 该合约被限流，未能抓到
            continue
        if df is not None and not df.empty:
            frames.append(df)
            seen_real = True
            consecutive_empty = 0
        else:
            if seen_real:
                consecutive_empty += 1
                if consecutive_empty >= stop_after_empty:
                    break
    if not frames:
        return pd.DataFrame(), complete
    return build_weighted(frames, min_dt), complete


def main():
    parser = argparse.ArgumentParser(description="下载期货全品种 15 分钟加权数据")
    parser.add_argument("--products", type=str, default=None,
                        help="指定品种(逗号分隔, 如 RB,M,CU)，默认全品种")
    parser.add_argument("--period", type=int, default=15,
                        choices=[1, 5, 15, 30, 60], help="K线周期(分钟)")
    parser.add_argument("--rate", type=float, default=0.4,
                        help="全局最小请求间隔(秒)，防反爬封禁")
    parser.add_argument("--window-days", type=int, default=75,
                        help="保留最近 N 天的加权数据")
    parser.add_argument("--back-months", type=int, default=3,
                        help="向前回溯多少个月的到期合约(扩历史用)")
    parser.add_argument("--forward-months", type=int, default=19,
                        help="向后探测多少个月的远月合约")
    parser.add_argument("--stop-after-empty", type=int, default=4,
                        help="找到首个真实合约后, 连续N个月空缺则停止探测远月(扩长历史时调大防截断)")
    parser.add_argument("--retry-passes", type=int, default=4,
                        help="对无数据/不完整品种额外重试的轮数(每轮前冷却)")
    parser.add_argument("--cooldown", type=float, default=30.0,
                        help="每轮重试前的冷却秒数")
    parser.add_argument("--outdir", type=str, default="data_futures",
                        help="输出根目录")
    args = parser.parse_args()

    if args.products:
        products = [p.strip().upper() for p in args.products.split(",") if p.strip()]
    else:
        products = list(PRODUCTS.keys())

    months = candidate_months(args.back_months, args.forward_months)
    min_dt = pd.Timestamp(datetime.now() - timedelta(days=args.window_days))
    out_root = Path(args.outdir) / f"{args.period}min"
    out_root.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; backtest-data)"})
    limiter = RateLimiter(args.rate)

    print(f"开始下载 {len(products)} 个品种的 {args.period} 分钟加权数据")
    print(f"候选月份: {months[0]} ~ {months[-1]} ({len(months)} 个)")
    print(f"窗口: 最近 {args.window_days} 天 (>= {min_dt:%Y-%m-%d})")
    print(f"顺序抓取  限速 {args.rate}s/req  输出 {out_root}")
    print("-" * 64)

    results: dict[str, dict] = {}      # product -> manifest row
    pending = list(products)           # 待抓取(无数据或不完整)

    for pass_no in range(args.retry_passes + 1):
        if not pending:
            break
        if pass_no > 0:
            print(f"\n第 {pass_no} 轮重试 {len(pending)} 个品种(无数据/不完整), "
                  f"冷却 {args.cooldown}s ...", flush=True)
            time.sleep(args.cooldown)

        retry_next = []
        for product in pending:
            exchange, name = PRODUCTS.get(product, ("?", product))
            try:
                df, complete = download_product(product, args.period, months,
                                                min_dt, session, limiter,
                                                args.stop_after_empty)
            except Exception as exc:
                print(f"❌ {product:4s} {name:8s} 异常: {exc}", flush=True)
                retry_next.append(product)
                continue

            if df.empty:
                # 完全无数据: complete=True 说明确实是停牌/无此品种，不再重试
                if complete:
                    print(f"⚠️  {product:4s} {name:8s} 无数据(已停牌/无在交易合约)", flush=True)
                else:
                    print(f"⏳ {product:4s} {name:8s} 被限流，稍后重试", flush=True)
                    retry_next.append(product)
                continue

            df.to_parquet(out_root / f"{product}.parquet", index=False)
            results[product] = {
                "product": product, "exchange": exchange, "name": name,
                "rows": len(df), "start": str(df["datetime"].iloc[0]),
                "end": str(df["datetime"].iloc[-1]),
                "max_contracts": int(df["n_contracts"].max()),
                "complete": complete,
            }
            flag = "" if complete else "  ⚠不完整(将重试)"
            print(f"✅ {product:4s} {name:8s} {len(df):5d} 行  "
                  f"{df['datetime'].iloc[0]:%Y-%m-%d} ~ {df['datetime'].iloc[-1]:%Y-%m-%d}  "
                  f"(最多 {int(df['n_contracts'].max())} 合约){flag}", flush=True)
            if not complete:
                retry_next.append(product)   # 落地了部分数据，但仍重试求完整
        pending = retry_next

    if results:
        man_df = pd.DataFrame(list(results.values())).sort_values(["exchange", "product"])
        man_path = Path(args.outdir) / f"manifest_{args.period}min.csv"
        man_df.to_csv(man_path, index=False)
        print("-" * 64)
        print(f"清单已写入: {man_path}")

    print("-" * 64)
    print(f"完成! 成功 {len(results)} 个品种, 无数据 {len(pending)} 个, "
          f"总计 {len(products)} 个")
    if pending:
        print(f"仍无数据: {', '.join(pending)}")


if __name__ == "__main__":
    main()
