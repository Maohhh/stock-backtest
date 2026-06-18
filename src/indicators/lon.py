"""
LON 长线指标 (龙系长线 / 通达信 LON)

LON 是一种量价结合的中长线趋势指标。其核心思想是用「成交量 / 振幅」对每日
涨跌进行加权累加，再对累加序列做两条不同周期的平滑，取差值作为多空分界：

    VID  = SUM(VOL, 2) / ((HHV(HIGH, 2) - LLV(LOW, 2)) * 100)   # 量幅比
    RC   = (CLOSE - REF(CLOSE, 1)) * VID                        # 加权涨跌
    LONG = SUM(RC, 0)                                           # 累加（从首根起）
    DIFF = SMA(LONG, 10, 1)                                     # 快平滑
    DEA  = SMA(LONG, 20, 1)                                     # 慢平滑
    LON  = DIFF - DEA

判读：
    LON > 0  快线在慢线上方，处于上涨趋势（0 轴上方）。
    LON < 0  快线在慢线下方，处于下跌趋势（0 轴下方）。

其中 SMA(X, N, 1) 为通达信式平滑：Y_t = (X_t + (N-1) * Y_{t-1}) / N，
等价于 alpha = 1/N 的指数平滑。
"""

import numpy as np
import pandas as pd


def _sma_cn(series: pd.Series, n: int, m: int = 1) -> pd.Series:
    """通达信 SMA(X, N, M): Y_t = (M*X_t + (N-M)*Y_{t-1}) / N，等价 alpha=M/N 的 EMA。"""
    return series.ewm(alpha=m / n, adjust=False).mean()


def lon(df: pd.DataFrame, fast: int = 10, slow: int = 20) -> pd.DataFrame:
    """
    计算 LON 长线指标。

    参数:
        df: 含 high / low / close / volume 列的 DataFrame
        fast: 快平滑周期，默认 10
        slow: 慢平滑周期，默认 20

    返回:
        含 LONG / DIFF / DEA / LON 四列的 DataFrame，索引与 df 对齐。
    """
    for col in ("high", "low", "close", "volume"):
        if col not in df.columns:
            raise ValueError(f"列 '{col}' 不存在于DataFrame中")

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    # 量幅比 VID = 两日成交量之和 / (两日振幅 * 100)
    vol2 = volume.rolling(window=2, min_periods=1).sum()
    rng2 = high.rolling(window=2, min_periods=1).max() - low.rolling(window=2, min_periods=1).min()
    denom = rng2 * 100.0
    vid = vol2 / denom.replace(0, np.nan)
    vid = vid.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 加权涨跌并累加
    rc = (close - close.shift(1)) * vid
    rc = rc.fillna(0.0)
    long_line = rc.cumsum()

    diff = _sma_cn(long_line, fast, 1)
    dea = _sma_cn(long_line, slow, 1)
    lon_val = diff - dea

    return pd.DataFrame(
        {
            "LONG": long_line,
            "DIFF": diff,
            "DEA": dea,
            "LON": lon_val,
        },
        index=df.index,
    )
