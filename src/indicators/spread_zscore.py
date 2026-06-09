"""
价差 Z 分数回归指标 (Spread Z-Score Mean-Reversion)

针对**协整的农产品配对**（典型：玉米 C 与玉米淀粉 CS——淀粉由玉米加工而来，
加工利润价差受套利约束强烈均值回归）设计的高胜率指标。

核心：把两腿价格合成一条"价差"序列，对其做滚动标准化得到 Z 分数；
Z 偏离过大时反向押注其回归。均值回归天生高胜率，配合 σ 止损控制盈亏比。

其他适用的同类农产品配对：豆粕-菜粕(M/RM)、豆油-菜油(Y/OI) 等替代/加工关系品种。

可视化：Z 分数本身就是一条振荡指标线，配 ±entry / exit / ±stop 几条横线即成主图/副图指标。
"""

import numpy as np
import pandas as pd


def build_spread(price_a: pd.Series, price_b: pd.Series, mode: str = "diff",
                 beta: float | None = None) -> pd.Series:
    """
    合成价差序列（b 相对 a）。

    参数:
        price_a, price_b: 两腿收盘价（已对齐、已清洗）
        mode: 'diff'  -> b - beta*a（默认 beta=1，适合同合约乘数，如 C/CS 都是 10t/手）
              'ratio' -> log(b) - log(a)（适合价位差异大的品种）
        beta: diff 模式下的对冲比例，None 则取 1.0
    """
    df = pd.concat({"a": price_a, "b": price_b}, axis=1).dropna()
    if mode == "ratio":
        return np.log(df["b"]) - np.log(df["a"])
    return df["b"] - (1.0 if beta is None else beta) * df["a"]


def spread_zscore(spread: pd.Series, window: int = 20) -> pd.Series:
    """价差的滚动 Z 分数：(spread - 均值) / 标准差。"""
    ma = spread.rolling(window).mean()
    sd = spread.rolling(window).std()
    return (spread - ma) / sd


def efficiency_ratio(spread: pd.Series, n: int = 10) -> pd.Series:
    """
    价差的 Kaufman 效率系数（净位移/路程，0~1）。
    越接近 1 = 单边趋势（价差在结构性走偏）；越接近 0 = 来回震荡（适合做回归）。
    """
    net = (spread - spread.shift(n)).abs()
    path = spread.diff().abs().rolling(n).sum()
    return net / path


def reversion_position(z: pd.Series, entry: float = 1.5, exit: float = 0.3,
                       stop: float = 4.0, er: pd.Series | None = None,
                       er_max: float | None = None) -> pd.Series:
    """
    由 Z 分数生成价差持仓信号（状态机），可选**趋势过滤**。

    +1 = 做多价差（买 b 卖 a）；-1 = 做空价差（卖 b 买 a）；0 = 空仓。

    规则:
        - Z < -entry  -> 开多（价差被低估，赌回升）
        - Z >  entry  -> 开空（价差被高估，赌回落）
        - 多头: Z 回到 >= -exit 平仓止盈；Z < -stop 止损
        - 空头: Z 回到 <=  exit 平仓止盈；Z >  stop 止损
        - 趋势过滤: 若给定 er/er_max，当 ER > er_max（价差处于单边趋势）时**不开新仓**，
          避免在趋势行情里反复抄底/摸顶被止损（已持仓的止盈止损不受影响）。

    返回:
        与 z 同索引的持仓 Series（取值 -1/0/+1），可直接 shift(1) 后用于回测。
    """
    pos = np.zeros(len(z))
    state = 0
    zv = z.values
    ev = er.values if er is not None else None
    for i in range(len(z)):
        if np.isnan(zv[i]):
            pos[i] = state
            continue
        if state == 0:
            trending = (ev is not None and er_max is not None
                        and not np.isnan(ev[i]) and ev[i] > er_max)
            if not trending:
                if zv[i] < -entry:
                    state = 1
                elif zv[i] > entry:
                    state = -1
        elif state == 1:                       # 持多价差
            if zv[i] >= -exit or zv[i] < -stop:
                state = 0
        elif state == -1:                      # 持空价差
            if zv[i] <= exit or zv[i] > stop:
                state = 0
        pos[i] = state
    return pd.Series(pos, index=z.index)
