"""
期货多周期数据模块
==================

从新浪财经获取国内商品期货 **主力连续合约** 的日线与分钟线数据，并提供
多周期（周/月/自定义）重采样工具。这是 "看大周期做小周期" 多周期回测的
数据基础。

数据说明
--------
- 日线接口 ``getDailyKLine`` 返回主力连续合约的全部历史（部分品种可回溯到
  2005 年），字段：date / open / high / low / close / volume / open_interest /
  settle。
- 分钟接口 ``getFewMinLine`` 只返回最近约 1000 根 K 线（约 2 个月），适合做
  日内多周期的演示，不适合长周期年化统计。
- 主力连续合约由新浪按主力切换 **直接拼接**（未做后复权），换月处会有跳空。
  对趋势类策略影响有限，但做收益率统计时需知悉该局限，本模块用收盘价的
  百分比收益来计算策略损益，避免跳空带来的绝对价差失真。

品种按底层逻辑分为 5 大板块（农产品 / 黑色 / 有色 / 贵金属 / 能化），不同
板块的驱动逻辑不同，回测时分板块统计更有意义。
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

_UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
}

# 主力连续合约代码以 "0" 结尾，例如 RB0 = 螺纹钢主力连续。
# 分板块组织：每个品种是 (symbol, 中文名)。
FUTURES_UNIVERSE: Dict[str, List[Tuple[str, str]]] = {
    "黑色": [
        ("RB0", "螺纹钢"), ("I0", "铁矿石"), ("HC0", "热卷"),
        ("J0", "焦炭"), ("JM0", "焦煤"), ("SM0", "锰硅"), ("SF0", "硅铁"),
    ],
    "有色": [
        ("CU0", "沪铜"), ("AL0", "沪铝"), ("ZN0", "沪锌"),
        ("NI0", "沪镍"), ("SN0", "沪锡"), ("PB0", "沪铅"),
    ],
    "贵金属": [
        ("AU0", "黄金"), ("AG0", "白银"),
    ],
    "农产品": [
        ("M0", "豆粕"), ("Y0", "豆油"), ("C0", "玉米"), ("A0", "豆一"),
        ("P0", "棕榈油"), ("CF0", "棉花"), ("SR0", "白糖"),
        ("OI0", "菜油"), ("RM0", "菜粕"), ("CS0", "淀粉"), ("B0", "豆二"),
    ],
    "能化": [
        ("TA0", "PTA"), ("MA0", "甲醇"), ("PP0", "聚丙烯"), ("L0", "塑料"),
        ("V0", "PVC"), ("RU0", "橡胶"), ("BU0", "沥青"),
        ("FG0", "玻璃"), ("EG0", "乙二醇"), ("SA0", "纯碱"),
    ],
}


def all_symbols() -> List[str]:
    return [s for syms in FUTURES_UNIVERSE.values() for s, _ in syms]


def symbol_category(symbol: str) -> Optional[str]:
    for cat, syms in FUTURES_UNIVERSE.items():
        if any(s == symbol for s, _ in syms):
            return cat
    return None


def symbol_name(symbol: str) -> str:
    for syms in FUTURES_UNIVERSE.values():
        for s, name in syms:
            if s == symbol:
                return name
    return symbol


class FuturesDataSource:
    """新浪期货主力连续合约数据源（带本地缓存）。"""

    DAILY_URL = (
        "https://stock2.finance.sina.com.cn/futures/api/jsonp.php/"
        "var%20_t={symbol}/InnerFuturesNewService.getDailyKLine?symbol={symbol}"
    )
    MIN_URL = (
        "https://stock2.finance.sina.com.cn/futures/api/jsonp.php/"
        "var%20_t={symbol}_{period}/InnerFuturesNewService.getFewMinLine"
        "?symbol={symbol}&type={period}"
    )

    def __init__(self, cache_dir: str = "data/futures", use_cache: bool = True):
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        (self.cache_dir / "daily").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "minute").mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(_UA)

    # ------------------------------------------------------------------ #
    # 网络请求
    # ------------------------------------------------------------------ #
    def _get_json_array(self, url: str, retries: int = 3) -> Optional[list]:
        last_err = None
        for attempt in range(retries):
            try:
                resp = self.session.get(url, timeout=20)
                resp.raise_for_status()
                m = re.search(r"\((\[.*\])\)", resp.text, re.S)
                if not m:
                    return None
                return json.loads(m.group(1))
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        if last_err:
            print(f"  ⚠️ 请求失败 {url[:80]}... : {last_err!r}")
        return None

    # ------------------------------------------------------------------ #
    # 日线
    # ------------------------------------------------------------------ #
    def get_daily(self, symbol: str, force: bool = False) -> Optional[pd.DataFrame]:
        cache = self.cache_dir / "daily" / f"{symbol}.parquet"
        if self.use_cache and not force and cache.exists():
            return pd.read_parquet(cache)

        url = self.DAILY_URL.format(symbol=symbol)
        data = self._get_json_array(url)
        if not data:
            return None

        df = pd.DataFrame(data)
        rename = {"d": "date", "o": "open", "h": "high", "l": "low",
                  "c": "close", "v": "volume", "p": "open_interest", "s": "settle"}
        df = df.rename(columns=rename)
        for col in ("open", "high", "low", "close", "volume", "open_interest", "settle"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
        df.attrs["symbol"] = symbol
        if self.use_cache:
            df.to_parquet(cache, index=False)
        return df

    def get_minute(self, symbol: str, period: int = 15,
                   force: bool = False) -> Optional[pd.DataFrame]:
        """period 取值：5 / 15 / 30 / 60（分钟）。"""
        cache = self.cache_dir / "minute" / f"{symbol}_{period}.parquet"
        if self.use_cache and not force and cache.exists():
            return pd.read_parquet(cache)

        url = self.MIN_URL.format(symbol=symbol, period=period)
        data = self._get_json_array(url)
        if not data:
            return None

        df = pd.DataFrame(data)
        rename = {"d": "datetime", "o": "open", "h": "high", "l": "low",
                  "c": "close", "v": "volume", "p": "open_interest"}
        df = df.rename(columns=rename)
        for col in ("open", "high", "low", "close", "volume", "open_interest"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.dropna(subset=["close"]).sort_values("datetime").reset_index(drop=True)
        df.attrs["symbol"] = symbol
        if self.use_cache:
            df.to_parquet(cache, index=False)
        return df

    def get_universe_daily(self, force: bool = False) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for symbol in all_symbols():
            df = self.get_daily(symbol, force=force)
            if df is not None and len(df) > 250:
                out[symbol] = df
        return out


# ---------------------------------------------------------------------- #
# 多周期重采样
# ---------------------------------------------------------------------- #
_AGG = {
    "open": "first", "high": "max", "low": "min", "close": "last",
    "volume": "sum", "open_interest": "last", "settle": "last",
}


def resample_ohlc(df: pd.DataFrame, rule: str,
                  date_col: str = "date") -> pd.DataFrame:
    """把日线/分钟线重采样到更大的周期。

    rule 例：'W'（周）、'ME'（月）、'2W'、'4H' 等 pandas offset alias。
    """
    g = df.set_index(date_col)
    agg = {k: v for k, v in _AGG.items() if k in g.columns}
    out = g.resample(rule, label="right", closed="right").agg(agg).dropna(subset=["close"])
    out = out.reset_index().rename(columns={date_col: date_col})
    return out


if __name__ == "__main__":
    src = FuturesDataSource()
    rb = src.get_daily("RB0")
    print("RB0 日线:", None if rb is None else (len(rb), rb["date"].min(), rb["date"].max()))
    if rb is not None:
        wk = resample_ohlc(rb, "W")
        print("RB0 周线:", len(wk), wk.tail(2)[["date", "close"]].to_dict("records"))
