"""
新浪期货日线数据获取模块

通过新浪期货行情接口获取单个期货合约（含已交割的历史合约）以及主力连续
合约的日线数据，用于跨期 / 跨品种价差套利回测。

接口返回字段：
    d 日期, o 开盘, h 最高, l 最低, c 收盘, v 成交量, p 持仓量, s 结算价

具体合约示例：``RM2509``（郑商所菜粕 2509）、``jd2509``（大商所鸡蛋 2509）。
主力连续示例：``TA0``（PTA 主力连续）、``cs0``（玉米淀粉主力连续）。
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

_API = (
    "https://stock2.finance.sina.com.cn/futures/api/jsonp.php/"
    "var%20_=/InnerFuturesNewService.getDailyKLine?symbol={symbol}"
)
_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://finance.sina.com.cn",
}

_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "futures"


def _parse(payload: str) -> pd.DataFrame:
    start = payload.find("([")
    if start < 0:
        return pd.DataFrame()
    end = payload.rfind(")")
    rows = json.loads(payload[start + 1 : end])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.rename(
        columns={
            "d": "date",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "p": "open_interest",
            "s": "settle",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume", "open_interest", "settle"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def fetch_daily(
    symbol: str,
    use_cache: bool = True,
    retries: int = 4,
    timeout: int = 25,
) -> pd.DataFrame:
    """获取单个合约 / 主力连续的全部日线数据。

    Parameters
    ----------
    symbol : str
        合约代码，如 ``RM2509`` 或主力连续 ``TA0``。
    use_cache : bool
        是否使用本地 CSV 缓存（``data/futures/<symbol>.csv``）。
        已交割合约数据不再变化，命中缓存可避免重复请求。
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE_DIR / f"{symbol}.csv"
    # 主力连续（以 0 结尾）每天都在更新，缓存当天有效即可；
    # 具体月份合约一旦有数据基本不再变化，长期可用缓存。
    is_continuous = symbol.endswith("0")
    if use_cache and cache_file.exists():
        fresh = (time.time() - cache_file.stat().st_mtime) < 12 * 3600
        if not is_continuous or fresh:
            df = pd.read_csv(cache_file, parse_dates=["date"])
            if not df.empty:
                return df

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(_API.format(symbol=symbol), headers=_HEADERS)
            raw = urllib.request.urlopen(req, timeout=timeout).read().decode("gbk", "ignore")
            df = _parse(raw)
            if not df.empty and use_cache:
                df.to_csv(cache_file, index=False)
            return df
        except Exception as exc:  # noqa: BLE001 - 网络异常统一重试
            last_err = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"获取 {symbol} 数据失败: {last_err}")


def fetch_close(symbol: str, **kwargs) -> pd.Series:
    """返回以日期为索引的收盘价序列。"""
    df = fetch_daily(symbol, **kwargs)
    if df.empty:
        return pd.Series(dtype=float, name=symbol)
    return df.set_index("date")["close"].rename(symbol)


def clean_spikes(s: pd.Series, max_jump: float) -> pd.Series:
    """剔除孤立的单日尖刺（主力连续拼接 / 异常报价导致的脏数据）。

    当某一日相对前一日跳变超过 ``max_jump``，且次日又反向跳回（绝对跳变同样
    超过 ``max_jump``）时，判定为孤立尖刺并删除该点。
    """
    s = s.sort_index()
    vals = s.values
    keep = [True] * len(vals)
    for i in range(1, len(vals) - 1):
        up = vals[i] - vals[i - 1]
        down = vals[i + 1] - vals[i]
        if abs(up) > max_jump and abs(down) > max_jump and (up > 0) != (down > 0):
            keep[i] = False
    return s[pd.Series(keep, index=s.index)]
