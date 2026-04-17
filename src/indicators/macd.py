"""
MACD 指标 (Moving Average Convergence Divergence)

指数平滑异同移动平均线，用于判断趋势和动量。
"""

import pandas as pd
import numpy as np
from .ma import ema


def macd(df: pd.DataFrame, 
         fast: int = 12, 
         slow: int = 26, 
         signal: int = 9,
         column: str = 'close') -> pd.DataFrame:
    """
    计算MACD指标
    
    MACD由三部分组成：
    - DIF (快线): 快速EMA减去慢速EMA
    - DEA (慢线/信号线): DIF的EMA
    - MACD柱状图: (DIF - DEA) * 2
    
    参数:
        df: 包含股票数据的DataFrame
        fast: 快速EMA周期，默认12
        slow: 慢速EMA周期，默认26
        signal: 信号线EMA周期，默认9
        column: 计算所用的价格列名，默认'close'
    
    返回:
        包含DIF、DEA、MACD柱状图的DataFrame
    
    示例:
        >>> result = macd(df)
        >>> result['DIF']  # 快线值
        >>> result['DEA']  # 慢线值
        >>> result['MACD']  # 柱状图
    """
    if column not in df.columns:
        raise ValueError(f"列 '{column}' 不存在于DataFrame中")
    
    # 计算快速和慢速EMA
    ema_fast = ema(df, period=fast, column=column)
    ema_slow = ema(df, period=slow, column=column)
    
    # DIF = 快速EMA - 慢速EMA
    dif = ema_fast - ema_slow
    
    # DEA = DIF的EMA
    dea = dif.ewm(span=signal, adjust=False, min_periods=1).mean()
    
    # MACD柱状图 = (DIF - DEA) * 2
    macd_hist = (dif - dea) * 2
    
    return pd.DataFrame({
        'DIF': dif,
        'DEA': dea,
        'MACD': macd_hist
    })
