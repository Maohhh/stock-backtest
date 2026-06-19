"""
LON + MACD 反转策略（v2，抄底/摸顶版，支持做多 / 做空）

与趋势跟随版 (lon_macd_strategy) 相反，本策略在趋势衰竭处逆势捕捉拐点：

    做多：
        - LON < 0（仍处下跌区），但 |LON| 在缩小（LON 从负值向 0 轴回升 = 跌势衰竭）；
        - MACD 两条线 DIF/DEA 都在 0 轴下方，且此刻完成金叉（DIF 上穿 DEA）；
        - 满足时做多（抄底）。
        - 平仓：MACD 柱状值 (DIF-DEA)*2 开始变小（多头动能见顶回落）。
    做空（镜像）：
        - LON > 0（仍处上涨区），但 |LON| 在缩小（LON 从正值向 0 轴回落 = 涨势衰竭）；
        - MACD 两条线 DIF/DEA 都在 0 轴上方，且此刻完成死叉（DIF 下穿 DEA）；
        - 满足时做空（摸顶）。
        - 平仓：MACD 柱状值开始变大（空头动能见底回升）。

返回带 state 列（+1 多 / -1 空 / 0 空仓）的 DataFrame，信号收盘确认、下一根 K 线建仓。
"""

import numpy as np
import pandas as pd

from ..indicators import macd, lon


def generate_signals(
    df: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    lon_fast: int = 10,
    lon_slow: int = 20,
    warmup: int = 60,
    allow_reverse: bool = True,
    exit_mode: str = "hist_turn",
    exit_confirm: int = 1,
) -> pd.DataFrame:
    """
    计算指标并生成反转策略的多空状态序列。

    参数:
        df: 含 open/high/low/close/volume 的日线 DataFrame（按时间升序）
        macd_*: MACD 参数
        lon_*: LON 参数
        warmup: 预热 K 线数，之前不开仓
        allow_reverse: 平仓当根若出现反向开仓信号，是否直接反手
        exit_mode: 离场方式
            "hist_turn" —— 用户原版：MACD 柱状值反向（连续 exit_confirm 根）即平仓
            "cross"     —— 持有到反向交叉：多头等死叉、空头等金叉才平仓（更宽松）
        exit_confirm: hist_turn 模式下，柱状值需连续反向多少根才离场（默认 1）

    返回:
        在原 df 基础上追加 dif/dea/hist/lon/state 等列的新 DataFrame
    """
    data = df.reset_index(drop=True).copy()

    macd_df = macd(data, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    data["dif"] = macd_df["DIF"].values
    data["dea"] = macd_df["DEA"].values
    data["hist"] = macd_df["MACD"].values  # 柱状图 = (DIF-DEA)*2
    lon_df = lon(data, fast=lon_fast, slow=lon_slow)
    data["lon"] = lon_df["LON"].values

    dif = data["dif"]
    dea = data["dea"]
    hist = data["hist"]
    prev_dif = dif.shift(1)
    prev_dea = dea.shift(1)

    golden_cross = (dif > dea) & (prev_dif <= prev_dea)
    death_cross = (dif < dea) & (prev_dif >= prev_dea)

    lon_val = data["lon"]
    lon_rising = lon_val > lon_val.shift(1)   # LON 上升（在负区即 |LON| 缩小）
    lon_falling = lon_val < lon_val.shift(1)  # LON 下降（在正区即 |LON| 缩小）

    # 做多：下跌区跌势衰竭 + 0 轴下方金叉
    long_entry = (lon_val < 0) & lon_rising & golden_cross & (dif < 0) & (dea < 0)
    # 做空：上涨区涨势衰竭 + 0 轴上方死叉
    short_entry = (lon_val > 0) & lon_falling & death_cross & (dif > 0) & (dea > 0)

    # 平仓
    if exit_mode == "cross":
        long_exit = death_cross    # 多头持有到死叉
        short_exit = golden_cross  # 空头持有到金叉
    else:  # hist_turn：柱状动能连续 exit_confirm 根反向
        hist_down = hist < hist.shift(1)
        hist_up = hist > hist.shift(1)
        long_exit = hist_down.rolling(window=exit_confirm).sum() == exit_confirm
        short_exit = hist_up.rolling(window=exit_confirm).sum() == exit_confirm

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
