"""
移动平均线指标 (Moving Average)

提供简单移动平均 (SMA) 和指数移动平均 (EMA) 计算。
"""

import pandas as pd
import numpy as np


def sma(df: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.Series:
    """
    计算简单移动平均线 (Simple Moving Average)
    
    参数:
        df: 包含股票数据的DataFrame
        period: 移动平均周期，默认20
        column: 计算所用的价格列名，默认'close'
    
    返回:
        包含SMA值的Series
    
    示例:
        >>> sma(df, period=20)  # 计算20日简单移动平均
    """
    if column not in df.columns:
        raise ValueError(f"列 '{column}' 不存在于DataFrame中")
    
    return df[column].rolling(window=period, min_periods=1).mean()


def ema(df: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.Series:
    """
    计算指数移动平均线 (Exponential Moving Average)
    
    EMA给予近期价格更高的权重，比SMA对价格变化更敏感。
    
    参数:
        df: 包含股票数据的DataFrame
        period: 移动平均周期，默认20
        column: 计算所用的价格列名，默认'close'
    
    返回:
        包含EMA值的Series
    
    示例:
        >>> ema(df, period=12)  # 计算12日指数移动平均
    """
    if column not in df.columns:
        raise ValueError(f"列 '{column}' 不存在于DataFrame中")
    
    return df[column].ewm(span=period, adjust=False, min_periods=1).mean()
