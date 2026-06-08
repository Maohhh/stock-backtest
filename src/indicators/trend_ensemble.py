"""
单标的趋势集成指标 (Trend Ensemble, single-instrument)

针对**单个品种**的方向性趋势指标，可直接落到一张 K 线图上（含同花顺主图公式）。
设计目标：在中国期货"日线偏震荡"的环境里，尽量只在**真正有趋势**时入场，
避开震荡期的来回打脸（whipsaw），从而做到"高年化不爆仓"里的"不爆仓"。

两个核心部件：
1. 多周期 EMA 交叉投票（8/32, 16/64, 32/128）—— 多个时间尺度共振才给强信号；
2. Kaufman 效率系数（Efficiency Ratio, ER）过滤 —— ER 低（震荡）时强制空仓。

经过 43 个品种、最长 17 年日线的样本外检验：该指标在**宏观驱动品种**
（黄金 AU 最优，样本外夏普≈0.88、铜/棉花次之）上稳健为正；
纯震荡品种上接近 0——这与"趋势策略只在趋势品种上有效"的常识一致。
"""

import numpy as np
import pandas as pd

DEFAULT_PAIRS = [(8, 32), (16, 64), (32, 128)]


def efficiency_ratio(close: pd.Series, n: int = 20) -> pd.Series:
    """
    Kaufman 效率系数：n 日净位移 / n 日路程总和，取值 0~1。
    接近 1 = 单边趋势；接近 0 = 来回震荡。
    """
    net = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n).sum()
    return net / path


def trend_signal(df: pd.DataFrame, pairs=DEFAULT_PAIRS, er_n: int = 20,
                 er_th: float = 0.3, column: str = "close") -> pd.Series:
    """
    计算单标的趋势信号，取值范围 [-1, 1]。

    - 多周期 EMA 快线-慢线之差取符号，多周期平均得到投票强度；
    - 当 ER ≤ er_th（震荡）时，信号置 0（空仓）。

    参数:
        df: 单品种 OHLC DataFrame
        pairs: [(快, 慢)] EMA 周期对列表
        er_n: 效率系数回看周期
        er_th: 效率系数阈值，低于此值视为震荡、空仓
        column: 价格列名（默认 close）

    返回:
        趋势信号 Series（+ 看多 / − 看空 / 0 空仓）
    """
    close = df[column]
    votes = sum(np.sign(close.ewm(span=f).mean() - close.ewm(span=s).mean())
                for f, s in pairs) / len(pairs)
    er = efficiency_ratio(close, er_n)
    return votes.where(er > er_th, 0.0)
