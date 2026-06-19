"""
LON + MACD 进场 + ATR 移动止损出场（吊灯出场 / Chandelier Exit）

进场沿用 v1 趋势版信号（与均线无关）：
    做多：LON>0 且 MACD 金叉且 DIF/DEA 均>0
    做空：LON<0 且 MACD 死叉且 DIF/DEA 均<0

出场改为 ATR 移动止损（让利润奔跑、按波动率控风险，而非固定点数）：
    多头：初始止损 = 进场收盘 - k*ATR；之后 止损 = max(旧止损, 持仓以来最高价 - k*ATR)；
          某根最低价跌破止损即离场。
    空头：镜像。

返回带 state 列（+1/-1/0）的 DataFrame，信号收盘确认、下一根 K 线生效（交由回测引擎滞后一根）。
"""

import numpy as np
import pandas as pd

from ..indicators import macd, lon, atr


def generate_signals(
    df: pd.DataFrame,
    k: float = 3.0,
    atr_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    lon_fast: int = 10,
    lon_slow: int = 20,
    warmup: int = 60,
) -> pd.DataFrame:
    """
    参数:
        df: 含 open/high/low/close/volume 的日线 DataFrame（升序）
        k: ATR 止损倍数（吊灯系数，常用 2~4）
        atr_period: ATR 周期
        warmup: 预热根数

    返回:
        追加 dif/dea/lon/atr/stop/state 列的新 DataFrame
    """
    data = df.reset_index(drop=True).copy()
    m = macd(data, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    dif, dea = m["DIF"], m["DEA"]
    pdif, pdea = dif.shift(1), dea.shift(1)
    golden = (dif > dea) & (pdif <= pdea)
    death = (dif < dea) & (pdif >= pdea)
    lv = lon(data, fast=lon_fast, slow=lon_slow)["LON"]
    data["dif"], data["dea"], data["lon"] = dif, dea, lv

    long_entry = ((lv > 0) & golden & (dif > 0) & (dea > 0)).fillna(False).to_numpy()
    short_entry = ((lv < 0) & death & (dif < 0) & (dea < 0)).fillna(False).to_numpy()
    a = atr(data, period=atr_period).to_numpy()

    high = data["high"].to_numpy()
    low = data["low"].to_numpy()
    close = data["close"].to_numpy()
    n = len(data)

    states = np.zeros(n, dtype=int)
    stops = np.full(n, np.nan)
    pos = 0
    ext = stop = np.nan
    for t in range(n):
        if t < warmup:
            states[t] = 0
            continue
        # 先用当根 K 线检查移动止损是否被打
        if pos == 1:
            ext = max(ext, high[t])
            stop = max(stop, ext - k * a[t])
            if low[t] <= stop:
                pos = 0
        elif pos == -1:
            ext = min(ext, low[t])
            stop = min(stop, ext + k * a[t])
            if high[t] >= stop:
                pos = 0
        # 空仓则按信号进场，初始止损按 k*ATR
        if pos == 0:
            if long_entry[t]:
                pos = 1
                ext = high[t]
                stop = close[t] - k * a[t]
            elif short_entry[t]:
                pos = -1
                ext = low[t]
                stop = close[t] + k * a[t]
        states[t] = pos
        stops[t] = stop if pos != 0 else np.nan

    data["atr"] = a
    data["stop"] = stops
    data["state"] = states
    return data
