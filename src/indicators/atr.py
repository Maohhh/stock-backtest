"""
ATR 指标 (Average True Range)

平均真实波幅，用于衡量市场波动性。
"""

import pandas as pd
import numpy as np


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    计算ATR平均真实波幅
    
    ATR用于衡量价格波动幅度，常用于：
    - 设置止损位
    - 判断市场波动性
    - 仓位管理
    
    真实波幅(TR)是以下三者的最大值：
    1. 当日最高价 - 当日最低价
    2. |当日最高价 - 前一日收盘价|
    3. |当日最低价 - 前一日收盘价|
    
    ATR = TR的N周期简单移动平均
    
    参数:
        df: 包含股票数据的DataFrame (需有high, low, close列)
        period: ATR计算周期，默认14
    
    返回:
        包含ATR值的Series
    
    示例:
        >>> atr_values = atr(df, period=14)
        >>> atr_values > atr_values.mean()  # 高波动期
    """
    required_cols = ['high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"列 '{col}' 不存在于DataFrame中，ATR需要high/low/close列")
    
    # 计算真实波幅 (True Range)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    
    # TR = max(high-low, |high-previous_close|, |low-previous_close|)
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # ATR = TR的N周期简单移动平均 (使用Wilder's smoothing)
    atr = tr.ewm(alpha=1/period, min_periods=1, adjust=False).mean()
    
    return atr
