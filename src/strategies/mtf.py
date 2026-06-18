"""
多周期 "看大做小" 策略（Multi-Timeframe Top-Down）
==================================================

核心思想（综合 "多周期共振 / 看大周期做小周期" 的主流做法）：

    大周期定方向，小周期找入场，只做与大周期同向的单。

- **大周期 (HTF)**：把日线重采样成周线，用 EMA 排列 + 斜率判定趋势方向
  ``bias ∈ {-1, 0, +1}``（多 / 震荡观望 / 空）。
- **小周期 (LTF)**：在日线上用突破 / 均线交叉 / MACD 产生入场信号。
- **共振过滤**：最终信号 = 小周期入场信号 **且方向与大周期一致**，否则空仓。
  这就是 "看大做小" —— 顺着大周期的势，用小周期择时，过滤掉逆势噪音。

所有信号都做了 **防未来函数** 处理：大周期指标取 *已完成* 的上一根 HTF K 线
（merge_asof backward + shift），日线信号用当根收盘、下一根生效（引擎里再 shift）。

不同板块底层逻辑不同，参数（趋势窗口、突破周期）可分板块微调，详见 runner。
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from src.data.futures import resample_ohlc


# --------------------------------------------------------------------- #
# 大周期方向 (HTF bias)
# --------------------------------------------------------------------- #
def htf_bias(
    daily: pd.DataFrame,
    rule: str = "W",
    ema_fast: int = 8,
    ema_slow: int = 20,
    use_slope: bool = True,
) -> pd.Series:
    """在更大的周期上判定趋势方向，返回对齐到日线 date 的 -1/0/+1 序列。"""
    htf = resample_ohlc(daily, rule).copy()
    ef = htf["close"].ewm(span=ema_fast, adjust=False).mean()
    es = htf["close"].ewm(span=ema_slow, adjust=False).mean()
    slope_up = es.diff() > 0
    slope_dn = es.diff() < 0

    long_ok = (ef > es) & (htf["close"] > es)
    short_ok = (ef < es) & (htf["close"] < es)
    if use_slope:
        long_ok &= slope_up
        short_ok &= slope_dn

    bias = pd.Series(0, index=htf.index, dtype=float)
    bias[long_ok] = 1.0
    bias[short_ok] = -1.0

    htf_bias_df = pd.DataFrame({"date": htf["date"], "bias": bias})
    # 防未来函数：用上一根已完成的 HTF K 线
    htf_bias_df["bias"] = htf_bias_df["bias"].shift(1)

    merged = pd.merge_asof(
        daily[["date"]].sort_values("date"),
        htf_bias_df.sort_values("date"),
        on="date",
        direction="backward",
    )
    out = merged["bias"].fillna(0.0)
    out.index = daily["date"].values
    return out


# --------------------------------------------------------------------- #
# 小周期入场 (LTF entry)
# --------------------------------------------------------------------- #
def ltf_signal(
    daily: pd.DataFrame,
    method: Literal["donchian", "ma_cross", "macd", "trend_ride"] = "donchian",
    donchian: int = 20,
    ma_fast: int = 10,
    ma_slow: int = 30,
    stop_period: int = 20,
) -> pd.Series:
    """日线上的入场信号，返回 -1/0/+1（持仓状态，带方向）。"""
    close = daily["close"].reset_index(drop=True)
    high = daily["high"].reset_index(drop=True)
    low = daily["low"].reset_index(drop=True)

    if method == "trend_ride":
        # 顺着大周期方向，用日线 EMA 作为 "在不在场" 的开关（看大做小的精髓）：
        # 价格在日线 EMA 上方→允许做多，下方→允许做空；EMA 充当移动止损/再入场线。
        ema = close.ewm(span=stop_period, adjust=False).mean()
        sig = pd.Series(0.0, index=close.index)
        sig[close > ema] = 1.0
        sig[close < ema] = -1.0

    elif method == "donchian":
        upper = high.rolling(donchian).max().shift(1)
        lower = low.rolling(donchian).min().shift(1)
        raw = pd.Series(np.nan, index=close.index)
        raw[close > upper] = 1.0
        raw[close < lower] = -1.0
        sig = raw.ffill().fillna(0.0)

    elif method == "ma_cross":
        f = close.rolling(ma_fast).mean()
        s = close.rolling(ma_slow).mean()
        sig = pd.Series(0.0, index=close.index)
        sig[f > s] = 1.0
        sig[f < s] = -1.0

    elif method == "macd":
        ef = close.ewm(span=12, adjust=False).mean()
        es = close.ewm(span=26, adjust=False).mean()
        dif = ef - es
        dea = dif.ewm(span=9, adjust=False).mean()
        sig = pd.Series(0.0, index=close.index)
        sig[dif > dea] = 1.0
        sig[dif < dea] = -1.0
    else:
        raise ValueError(f"unknown method {method}")

    sig.index = daily["date"].values
    return sig


# --------------------------------------------------------------------- #
# 多周期合成
# --------------------------------------------------------------------- #
def mtf_signal(
    daily: pd.DataFrame,
    rule: str = "W",
    ema_fast: int = 8,
    ema_slow: int = 20,
    ltf_method: str = "donchian",
    donchian: int = 20,
    ma_fast: int = 10,
    ma_slow: int = 30,
    stop_period: int = 20,
    mode: Literal["mtf", "ltf_only", "htf_only"] = "mtf",
) -> pd.Series:
    """生成最终目标信号 (-1/0/+1)。

    mode:
      - 'mtf'      : 看大做小（小周期入场 ∩ 大周期同向）—— 本策略主体
      - 'ltf_only' : 只用小周期入场（不看大周期）—— 对照组
      - 'htf_only' : 只持有大周期方向 —— 对照组
    """
    bias = htf_bias(daily, rule, ema_fast, ema_slow)
    ltf = ltf_signal(daily, ltf_method, donchian, ma_fast, ma_slow, stop_period)

    if mode == "htf_only":
        return bias
    if mode == "ltf_only":
        return ltf

    # mtf：方向一致才开仓，否则空仓
    out = pd.Series(0.0, index=ltf.index)
    long_ok = (ltf > 0) & (bias > 0)
    short_ok = (ltf < 0) & (bias < 0)
    out[long_ok] = 1.0
    out[short_ok] = -1.0
    return out
