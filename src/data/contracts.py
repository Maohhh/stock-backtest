"""
单合约（具体交割月）数据与价差重构

新浪期货提供具体合约（如 RB2510、C2509）的日线历史（含已退市合约，约回溯到 2018 年），
据此可**重构券商挂牌的套利指令**：
- 同月跨品种价差（跨品种套利，如 C&CS、M&RM）；
- 相邻月跨期价差（跨期套利，如 RB 近-远月）。

挂牌套利指令本身无免费历史行情，但用单腿重构在回测上完全等价。
"""

import os
import re
import glob
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from .futures import _fetch, _DAY_URL  # 复用新浪日线抓取

# 各品种主力交割月（用于枚举合约 / 配对）
CONTRACT_MONTHS: Dict[str, List[int]] = {
    "C": [1, 5, 9], "CS": [1, 5, 9], "M": [1, 5, 9], "RM": [1, 5, 9],
    "I": [1, 5, 9], "P": [1, 5, 9], "Y": [1, 5, 9], "OI": [1, 5, 9],
    "RB": [1, 5, 10], "HC": [1, 5, 10],
    "B": [1, 5, 9], "V": [1, 5, 9], "PP": [1, 5, 9], "L": [1, 5, 9],
    "CF": [1, 5, 9], "CY": [1, 5, 9], "SF": [1, 5, 9], "SM": [1, 5, 9],
    "FG": [1, 5, 9], "SA": [1, 5, 9],
}


def download_contract(symbol: str, out_dir: str = "data/futures_contracts") -> Optional[pd.Series]:
    """下载单个具体合约的日线收盘价并缓存。symbol 形如 'RB2510'。"""
    os.makedirs(out_dir, exist_ok=True)
    df = _fetch(_DAY_URL, symbol, {"symbol": symbol})
    if df is None or len(df) < 30:
        return None
    s = df["close"].rename(symbol)
    s.to_csv(os.path.join(out_dir, f"{symbol}.csv"))
    return s


def download_product_contracts(product: str, years=range(18, 27),
                               out_dir: str = "data/futures_contracts",
                               pause: float = 0.12) -> List[str]:
    """批量下载某品种各年主力月份合约。"""
    months = CONTRACT_MONTHS.get(product.upper(), [1, 5, 9])
    got = []
    for yy in years:
        for mm in months:
            sym = f"{product.upper()}{yy:02d}{mm:02d}"
            path = os.path.join(out_dir, f"{sym}.csv")
            if os.path.exists(path):
                got.append(sym); continue
            if download_contract(sym, out_dir) is not None:
                got.append(sym)
            time.sleep(pause)
    return got


def _load(sym: str, data_dir: str) -> Optional[pd.Series]:
    path = os.path.join(data_dir, f"{sym}.csv")
    if not os.path.exists(path):
        return None
    return pd.to_numeric(pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0], errors="coerce")


def list_contracts(product: str, data_dir: str = "data/futures_contracts") -> List[Tuple[str, int, int, pd.Timestamp]]:
    """返回 [(代码, 年, 月, 交割月末)]，按交割时间排序。"""
    out = []
    for f in glob.glob(os.path.join(data_dir, f"{product.upper()}[0-9][0-9][0-9][0-9].csv")):
        sym = os.path.basename(f)[:-4]
        m = re.match(rf"{product.upper()}(\d\d)(\d\d)$", sym)
        if not m:
            continue
        yy, mm = int(m.group(1)), int(m.group(2))
        deliv = pd.Timestamp(2000 + yy, mm, 1) + pd.offsets.MonthEnd(0)
        out.append((sym, yy, mm, deliv))
    return sorted(out, key=lambda x: x[3])


def matched_spread_segments(prod_a: str, prod_b: str, mode: str = "diff",
                            truncate_days: int = 20,
                            data_dir: str = "data/futures_contracts") -> List[pd.Series]:
    """
    重构**同月跨品种价差**（跨品种套利指令的真身），按每个交割月切成一段段价差序列。

    mode='diff'  -> b - a（同乘数品种）
    mode='ratio' -> (log b - log a) * 10000（单位 bp，适合价位差异大的对）
    截断每段近月交割前 truncate_days 天，规避交割扰动。
    """
    ca = {(y, m): s for s, y, m, d in list_contracts(prod_a, data_dir)}
    cb = {(y, m): s for s, y, m, d in list_contracts(prod_b, data_dir)}
    dmap = {(y, m): d for s, y, m, d in list_contracts(prod_a, data_dir)}
    segs = []
    for key in sorted(set(ca) & set(cb)):
        a, b = _load(ca[key], data_dir), _load(cb[key], data_dir)
        if a is None or b is None:
            continue
        df = pd.DataFrame({"a": a, "b": b}).dropna()
        df = df[df.index < dmap[key] - pd.Timedelta(days=truncate_days)]
        if len(df) < 40:
            continue
        if mode == "ratio":
            segs.append((np.log(df["b"]) - np.log(df["a"])) * 10000)
        else:
            segs.append(df["b"] - df["a"])
    return segs


def calendar_spread_segments(product: str, truncate_days: int = 20,
                             data_dir: str = "data/futures_contracts") -> List[pd.Series]:
    """重构**相邻交割月跨期价差**（近-远），按相邻合约对切段。"""
    c = list_contracts(product, data_dir)
    segs = []
    for i in range(len(c) - 1):
        a, b = _load(c[i][0], data_dir), _load(c[i + 1][0], data_dir)
        if a is None or b is None:
            continue
        df = pd.DataFrame({"near": a, "far": b}).dropna()
        df = df[df.index < c[i][3] - pd.Timedelta(days=truncate_days)]
        if len(df) >= 40:
            segs.append(df["near"] - df["far"])
    return segs
