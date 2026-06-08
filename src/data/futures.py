"""
期货数据模块

提供从新浪财经获取期货连续主力合约的 15 分钟 / 日线数据，
本地缓存为 CSV，并按板块组织品种，供横截面策略使用。

数据来源：新浪期货 InnerFuturesNewService
- 15 分钟：getFewMinLine（仅返回最近约 1000 根）
- 日线：getDailyKLine（可回溯至上市初期，多年历史）

注意：连续主力（如 'RB0'）已做后复权拼接，适合做趋势 / 横截面研究。
"""

import os
import re
import json
import time
from typing import Dict, List, Optional

import pandas as pd
import requests

_MIN_URL = "https://stock2.finance.sina.com.cn/futures/api/jsonp.php/=/InnerFuturesNewService.getFewMinLine"
_DAY_URL = "https://stock2.finance.sina.com.cn/futures/api/jsonp.php/var%20_x=/InnerFuturesNewService.getDailyKLine"
_HEADERS = {"User-Agent": "Mozilla/5.0"}

# 按板块组织的连续主力品种（新浪代码，后缀 0 表示连续主力）
SECTORS: Dict[str, List[str]] = {
    "ferrous": ["RB0", "HC0", "I0", "JM0", "SS0", "SF0", "SM0"],          # 黑色
    "base_metal": ["CU0", "AL0", "ZN0", "NI0", "SN0", "PB0"],            # 有色
    "energy_chem": ["TA0", "MA0", "PP0", "L0", "V0", "EG0", "FU0", "BU0", "SC0"],  # 能化
    "oilseed": ["M0", "Y0", "P0", "RM0", "OI0", "A0"],                   # 油脂油料
    "grain": ["C0", "CS0"],                                              # 谷物
    "soft": ["CF0", "SR0", "AP0", "JD0"],                                # 软商品
    "precious": ["AU0", "AG0"],                                          # 贵金属
    "index": ["IF0", "IH0", "IC0", "IM0"],                              # 股指
}


def symbol_sector_map() -> Dict[str, str]:
    """返回 {品种: 板块} 映射。"""
    return {s: sec for sec, syms in SECTORS.items() for s in syms}


def _parse_jsonp(text: str) -> Optional[list]:
    m = re.search(r"=\((\[.*\])\)", text, re.S)
    if not m:
        return None
    return json.loads(m.group(1))


def _fetch(url: str, symbol: str, params: dict, retries: int = 3) -> Optional[pd.DataFrame]:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20, headers=_HEADERS)
            data = _parse_jsonp(r.text)
            if not data:
                return None
            df = pd.DataFrame(data)
            df["d"] = pd.to_datetime(df["d"])
            for c in ("o", "h", "l", "c", "v"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.rename(columns={"d": "datetime", "o": "open", "h": "high",
                                    "l": "low", "c": "close", "v": "volume"})
            return df.set_index("datetime").sort_index()
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(2 ** attempt)
    return None


def download_15min(symbol: str) -> Optional[pd.DataFrame]:
    """下载单品种 15 分钟连续主力数据（最近约 1000 根）。"""
    return _fetch(_MIN_URL, symbol, {"symbol": symbol, "type": "15"})


def download_daily(symbol: str) -> Optional[pd.DataFrame]:
    """下载单品种日线连续主力数据（多年历史）。"""
    return _fetch(_DAY_URL, symbol, {"symbol": symbol})


def download_sector(sector: str, freq: str = "15min",
                    out_dir: str = "data/futures_15min",
                    pause: float = 0.3) -> Dict[str, pd.DataFrame]:
    """
    批量下载某板块全部品种并缓存为 CSV。

    参数:
        sector: SECTORS 中的板块名
        freq: '15min' 或 'daily'
        out_dir: 缓存目录
    返回:
        {品种: DataFrame}
    """
    if sector not in SECTORS:
        raise ValueError(f"未知板块 {sector}，可选: {list(SECTORS)}")
    os.makedirs(out_dir, exist_ok=True)
    fn = download_15min if freq == "15min" else download_daily
    out: Dict[str, pd.DataFrame] = {}
    for sym in SECTORS[sector]:
        df = fn(sym)
        if df is not None and len(df):
            df.to_csv(os.path.join(out_dir, f"{sym}.csv"))
            out[sym] = df
        time.sleep(pause)
    return out


def load_panel(symbols: List[str], data_dir: str = "data/futures_15min",
               column: str = "close") -> pd.DataFrame:
    """
    从缓存 CSV 读取多品种，对齐成一个价格面板（列=品种，行=时间戳）。

    仅保留所有品种都有数据的时间戳（inner join），保证横截面对齐。
    """
    series = {}
    for sym in symbols:
        path = os.path.join(data_dir, f"{sym}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, parse_dates=[0], index_col=0)
        # 兼容已重命名(close)与原始新浪列名(c)
        alias = {"close": "c", "open": "o", "high": "h", "low": "l", "volume": "v"}
        if column in df.columns:
            col = column
        elif alias.get(column) in df.columns:
            col = alias[column]
        else:
            raise KeyError(f"{path} 缺少列 {column!r}")
        series[sym] = pd.to_numeric(df[col], errors="coerce")
    if not series:
        raise FileNotFoundError(f"在 {data_dir} 未找到 {symbols} 的任何数据")
    return pd.DataFrame(series).sort_index().dropna(how="any")
