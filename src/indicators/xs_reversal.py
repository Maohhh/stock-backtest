"""
横截面反转指标 (Cross-Sectional Reversal)

针对同一板块内高度相关的期货品种，捕捉日内 / 短期"过度反应"后的均值回归：
某品种在最近 `form` 根 K 线相对板块均值涨得越多，越倾向于回落，反之亦然。

经济学依据：同板块品种受同一宏观/产业因子驱动，价格高度同步；
当某品种短期偏离板块共同走势（板块中性后的残差），该偏离往往会回归。
这本质上是板块内的统计套利 / 配对回归思想。

输出为"目标权重面板"：列=品种、行=时间戳，多空、单位总杠杆（绝对值之和=1），
正权重做多、负权重做空，板块内多空大致中性。
"""

import numpy as np
import pandas as pd


def cross_sectional_reversal(prices: pd.DataFrame, form: int = 4) -> pd.DataFrame:
    """
    计算板块中性横截面反转的原始信号（未归一化）。

    参数:
        prices: 价格面板（列=品种，行=时间戳），需已横截面对齐
        form: 形成期，用最近 form 根 K 线的对数收益衡量短期表现

    返回:
        与 prices 同形状的信号 DataFrame；正值=低估应做多，负值=高估应做空。
    """
    if prices.shape[1] < 2:
        raise ValueError("横截面反转至少需要 2 个品种")
    log_ret = np.log(prices).diff(form)
    # 板块中性：减去该时刻所有品种的平均表现，得到相对强弱（残差）
    relative = log_ret.sub(log_ret.mean(axis=1), axis=0)
    # 反转：相对涨得多 -> 负权重（做空），相对跌得多 -> 正权重（做多）
    return -relative


def target_weights(prices: pd.DataFrame, form: int = 4, rebalance: int = 24) -> pd.DataFrame:
    """
    将反转信号转为可交易的目标权重面板。

    - 每个时刻按绝对值归一化，使总杠杆（|w| 之和）= 1；
    - 每隔 `rebalance` 根 K 线才更新一次目标权重（其余时刻保持不变），
      以此控制换手率、规避高频再平衡被交易成本吞噬。

    参数:
        prices: 价格面板
        form: 形成期（K 线根数）
        rebalance: 再平衡间隔（K 线根数）

    返回:
        目标权重 DataFrame（与 prices 同形状），已前向填充。
    """
    sig = cross_sectional_reversal(prices, form=form)
    gross = sig.abs().sum(axis=1).replace(0, np.nan)
    w = sig.div(gross, axis=0).fillna(0.0)
    # 仅在再平衡时点更新，其余时点沿用上一次权重
    update = pd.Series(np.arange(len(prices)) % rebalance == 0, index=prices.index)
    return w.where(update, np.nan).ffill().fillna(0.0)
