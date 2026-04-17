"""
RSI 指标 (Relative Strength Index)

相对强弱指标，用于衡量价格变动的速度和幅度，判断超买超卖。
"""

import pandas as pd
import numpy as np


def rsi(df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
    """
    计算RSI相对强弱指标
    
    RSI取值范围0-100：
    - RSI > 70: 超买状态
    - RSI < 30: 超卖状态
    - RSI = 50: 多空平衡
    
    参数:
        df: 包含股票数据的DataFrame
        period: RSI计算周期，默认14
        column: 计算所用的价格列名，默认'close'
    
    返回:
        包含RSI值的Series (0-100)
    
    示例:
        >>> rsi_values = rsi(df, period=14)
        >>> rsi_values > 70  # 识别超买
    """
    if column not in df.columns:
        raise ValueError(f"列 '{column}' 不存在于DataFrame中")
    
    # 计算价格变化
    delta = df[column].diff()
    
    # 分离上涨和下跌
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    # 计算平均上涨和平均下跌 (使用Wilder's smoothing方法)
    avg_gain = gain.ewm(alpha=1/period, min_periods=1, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=1, adjust=False).mean()
    
    # 计算相对强度RS
    rs = avg_gain / avg_loss
    
    # 计算RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi
