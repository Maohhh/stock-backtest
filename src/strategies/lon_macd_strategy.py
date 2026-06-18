"""
LON + MACD 趋势跟随策略（支持做多 / 做空）

策略规则（用户描述）：
    多头：
        - LON 长线指标在 0 轴上方（LON > 0），代表上涨趋势；
        - 在上涨趋势中，MACD 金叉（DIF 上穿 DEA）且 DIF、DEA 两条线都在 0 轴上方时买入开多；
        - 连续 3 根 K 线收盘价跌破 20 日均线时平多。
    空头（反之）：
        - LON 长线指标在 0 轴下方（LON < 0），代表下跌趋势；
        - 在下跌趋势中，MACD 死叉（DIF 下穿 DEA）且 DIF、DEA 两条线都在 0 轴下方时卖出开空；
        - 连续 3 根 K 线收盘价站上 20 日均线时平空。

`generate_signals` 返回带有指标列和 `state` 列的 DataFrame：
    state = +1 表示该 K 线收盘时应持有多头，
    state = -1 表示应持有空头，
    state =  0 表示空仓。
信号在收盘确认，回测引擎按「下一根 K 线」建仓，避免使用未来数据。
"""

import numpy as np
import pandas as pd

from ..indicators import macd, sma, lon


def generate_signals(
    df: pd.DataFrame,
    ma_period: int = 20,
    exit_bars: int = 3,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    lon_fast: int = 10,
    lon_slow: int = 20,
    warmup: int = 60,
    allow_reverse: bool = True,
) -> pd.DataFrame:
    """
    计算指标并生成多空状态序列。

    参数:
        df: 含 open/high/low/close/volume 的日线 DataFrame（按时间升序）
        ma_period: 均线周期，默认 20
        exit_bars: 连续多少根 K 线突破均线后离场，默认 3
        macd_*: MACD 参数
        lon_*: LON 参数
        warmup: 预热 K 线数，之前不开仓（指标尚未稳定）
        allow_reverse: 离场当根若出现反向开仓信号，是否直接反手

    返回:
        在原 df 基础上追加 dif/dea/ma/lon/state 等列的新 DataFrame
    """
    data = df.reset_index(drop=True).copy()

    macd_df = macd(data, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    data["dif"] = macd_df["DIF"].values
    data["dea"] = macd_df["DEA"].values
    data["ma"] = sma(data, period=ma_period).values
    lon_df = lon(data, fast=lon_fast, slow=lon_slow)
    data["lon"] = lon_df["LON"].values

    dif = data["dif"]
    dea = data["dea"]
    prev_dif = dif.shift(1)
    prev_dea = dea.shift(1)

    golden_cross = (dif > dea) & (prev_dif <= prev_dea)
    death_cross = (dif < dea) & (prev_dif >= prev_dea)

    long_entry = (data["lon"] > 0) & golden_cross & (dif > 0) & (dea > 0)
    short_entry = (data["lon"] < 0) & death_cross & (dif < 0) & (dea < 0)

    below_ma = data["close"] < data["ma"]
    above_ma = data["close"] > data["ma"]
    # 连续 exit_bars 根收盘价都在均线一侧
    long_exit = below_ma.rolling(window=exit_bars).sum() == exit_bars
    short_exit = above_ma.rolling(window=exit_bars).sum() == exit_bars

    long_entry = long_entry.fillna(False).to_numpy()
    short_entry = short_entry.fillna(False).to_numpy()
    long_exit = long_exit.fillna(False).to_numpy()
    short_exit = short_exit.fillna(False).to_numpy()

    n = len(data)
    states = np.zeros(n, dtype=int)
    state = 0
    for t in range(n):
        if t < warmup:
            states[t] = 0
            continue
        if state == 0:
            if long_entry[t]:
                state = 1
            elif short_entry[t]:
                state = -1
        elif state == 1:
            if long_exit[t]:
                state = 0
                if allow_reverse and short_entry[t]:
                    state = -1
        elif state == -1:
            if short_exit[t]:
                state = 0
                if allow_reverse and long_entry[t]:
                    state = 1
        states[t] = state

    data["state"] = states
    return data
