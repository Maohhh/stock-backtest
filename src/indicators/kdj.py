"""
KDJ 指标 (随机指标 Stochastic Oscillator)

用于判断超买超卖和趋势转折点。
"""

import pandas as pd
import numpy as np


def kdj(df: pd.DataFrame, 
        n: int = 9, 
        m1: int = 3, 
        m2: int = 3) -> pd.DataFrame:
    """
    计算KDJ随机指标
    
    KDJ由三条线组成：
    - K线: 快速确认线，对价格变化敏感
    - D线: 慢速主干线，K线的平滑
    - J线: 方向敏感线，反映K、D的乖离程度
    
    取值范围：
    - K、D: 0-100
    - J: 可能超出0-100范围
    
    常用用法：
    - K、D > 80: 超买区
    - K、D < 20: 超卖区
    - 金叉 (K上穿D): 买入信号
    - 死叉 (K下穿D): 卖出信号
    
    参数:
        df: 包含股票数据的DataFrame (需有high, low, close列)
        n: RSV计算周期，默认9
        m1: K线平滑周期，默认3
        m2: D线平滑周期，默认3
    
    返回:
        包含K、D、J值的DataFrame
    
    示例:
        >>> kdj_values = kdj(df)
        >>> kdj_values['K']  # K值
        >>> kdj_values['D']  # D值
        >>> kdj_values['J']  # J值
    """
    required_cols = ['high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"列 '{col}' 不存在于DataFrame中，KDJ需要high/low/close列")
    
    # 计算n周期内的最低最高价
    low_list = df['low'].rolling(window=n, min_periods=1).min()
    high_list = df['high'].rolling(window=n, min_periods=1).max()
    
    # 计算RSV (未成熟随机值)
    # RSV = (当日收盘价 - n日内最低价) / (n日内最高价 - n日内最低价) * 100
    rsv = (df['close'] - low_list) / (high_list - low_list) * 100
    
    # 处理除零情况
    rsv = rsv.fillna(50)
    
    # 计算K值 (RSV的m1周期移动平均)
    k = rsv.ewm(com=m1-1, adjust=False, min_periods=1).mean()
    
    # 计算D值 (K的m2周期移动平均)
    d = k.ewm(com=m2-1, adjust=False, min_periods=1).mean()
    
    # 计算J值
    j = 3 * k - 2 * d
    
    return pd.DataFrame({
        'K': k,
        'D': d,
        'J': j
    })
