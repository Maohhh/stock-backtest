"""
ATR 策略 (Average True Range Strategy)

基于平均真实波幅的波动率突破策略：
- 当价格突破ATR通道上轨时买入（波动率扩张+趋势向上）
- 当价格跌破ATR通道下轨时卖出（波动率扩张+趋势向下）
"""

from typing import Dict
import pandas as pd
import numpy as np
from .base import BaseStrategy
from ..indicators.atr import atr
from ..indicators.ma import sma


class ATRStrategy(BaseStrategy):
    """
    ATR波动率突破策略
    
    策略逻辑：
    - 基于ATR构建波动率通道（中轨 ± ATR倍数）
    - 当价格突破上轨时，认为波动率扩张且趋势向上，产生买入信号
    - 当价格跌破下轨时，认为波动率扩张且趋势向下，产生卖出信号
    
    参数:
        atr_period: ATR计算周期，默认14
        ma_period: 中轨移动平均周期，默认20
        multiplier: ATR倍数，默认2.0
        use_sma: 是否使用SMA作为中轨，默认True（False则使用收盘价）
    """
    
    def __init__(self, atr_period: int = 14, ma_period: int = 20, 
                 multiplier: float = 2.0, use_sma: bool = True):
        super().__init__("ATRStrategy")
        self.atr_period = atr_period
        self.ma_period = ma_period
        self.multiplier = multiplier
        self.use_sma = use_sma
    
    def set_params(self, **kwargs):
        """设置策略参数"""
        super().set_params(**kwargs)
        if 'atr_period' in kwargs:
            self.atr_period = kwargs['atr_period']
        if 'ma_period' in kwargs:
            self.ma_period = kwargs['ma_period']
        if 'multiplier' in kwargs:
            self.multiplier = kwargs['multiplier']
        if 'use_sma' in kwargs:
            self.use_sma = kwargs['use_sma']
    
    def on_bar(self, context: Dict) -> Dict:
        """
        每个Bar调用一次，生成交易信号
        
        Args:
            context: 包含date, price, portfolio, data的字典
        
        Returns:
            交易信号字典或None
        """
        data = context['data']
        
        # 数据不足时无法计算
        min_periods = max(self.atr_period, self.ma_period) + 1
        if len(data) < min_periods:
            return None
        
        # 检查必需的列
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                return None
        
        # 计算ATR
        atr_values = atr(data, period=self.atr_period)
        
        # 计算中轨（SMA或收盘价）
        if self.use_sma:
            middle = sma(data, period=self.ma_period)
        else:
            middle = data['close']
        
        # 计算上下轨
        upper = middle + (atr_values * self.multiplier)
        lower = middle - (atr_values * self.multiplier)
        
        # 获取当前值
        current_close = data['close'].iloc[-1]
        current_upper = upper.iloc[-1]
        current_lower = lower.iloc[-1]
        
        if len(data) > 1:
            prev_close = data['close'].iloc[-2]
            prev_upper = upper.iloc[-2]
            prev_lower = lower.iloc[-2]
            
            # 价格突破上轨 - 买入信号（波动率扩张+趋势向上）
            if prev_close <= prev_upper and current_close > current_upper:
                return {'direction': 'buy', 'amount': 100}
            
            # 价格跌破下轨 - 卖出信号（波动率扩张+趋势向下）
            if prev_close >= prev_lower and current_close < current_lower:
                return {'direction': 'sell', 'amount': 100}
        
        return None
