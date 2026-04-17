"""
布林带指标 (Bollinger Bands)

用于衡量价格波动性和识别超买超卖状态。
"""

import pandas as pd
import numpy as np
from .ma import sma


def bollinger_bands(df: pd.DataFrame, 
                    period: int = 20, 
                    std_dev: float = 2.0,
                    column: str = 'close') -> pd.DataFrame:
    """
    计算布林带指标
    
    布林带由三条线组成：
    - 中轨: 简单移动平均线 (SMA)
    - 上轨: 中轨 + 标准差倍数
    - 下轨: 中轨 - 标准差倍数
    
    常用用法：
    - 价格触及上轨: 可能超买
    - 价格触及下轨: 可能超卖
    - 带宽收窄: 可能即将出现大波动
    
    参数:
        df: 包含股票数据的DataFrame
        period: 移动平均周期，默认20
        std_dev: 标准差倍数，默认2.0
        column: 计算所用的价格列名，默认'close'
    
    返回:
        包含中轨、上轨、下轨的DataFrame
    
    示例:
        >>> bb = bollinger_bands(df)
        >>> bb['middle']  # 中轨
        >>> bb['upper']   # 上轨
        >>> bb['lower']   # 下轨
    """
    if column not in df.columns:
        raise ValueError(f"列 '{column}' 不存在于DataFrame中")
    
    # 计算中轨 (SMA)
    middle = sma(df, period=period, column=column)
    
    # 计算标准差
    rolling_std = df[column].rolling(window=period, min_periods=1).std()
    
    # 计算上轨和下轨
    upper = middle + (rolling_std * std_dev)
    lower = middle - (rolling_std * std_dev)
    
    return pd.DataFrame({
        'middle': middle,
        'upper': upper,
        'lower': lower
    })
