"""
RSI 策略 (Relative Strength Index Strategy)

基于相对强弱指标的交易策略：
- RSI超卖(<30)时买入
- RSI超买(>70)时卖出
"""

from typing import Dict
import pandas as pd
from .base import BaseStrategy
from ..indicators.rsi import rsi


class RSIStrategy(BaseStrategy):
    """
    RSI相对强弱指标策略
    
    策略逻辑：
    - 当RSI < 超卖阈值(默认30)时，认为市场超卖，产生买入信号
    - 当RSI > 超买阈值(默认70)时，认为市场超买，产生卖出信号
    
    参数:
        period: RSI计算周期，默认14
        oversold: 超卖阈值，默认30
        overbought: 超买阈值，默认70
    """
    
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__("RSIStrategy")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.prev_rsi = None
    
    def set_params(self, **kwargs):
        """设置策略参数"""
        super().set_params(**kwargs)
        if 'period' in kwargs:
            self.period = kwargs['period']
        if 'oversold' in kwargs:
            self.oversold = kwargs['oversold']
        if 'overbought' in kwargs:
            self.overbought = kwargs['overbought']
    
    def on_bar(self, context: Dict) -> Dict:
        """
        每个Bar调用一次，生成交易信号
        
        Args:
            context: 包含date, price, portfolio, data的字典
        
        Returns:
            交易信号字典或None
        """
        data = context['data']
        
        # 数据不足时无法计算RSI
        if len(data) < self.period + 1:
            return None
        
        # 计算RSI
        rsi_values = rsi(data, period=self.period)
        current_rsi = rsi_values.iloc[-1]
        
        # 需要前一天的RSI来判断是否穿越阈值
        if len(data) > 1:
            prev_rsi = rsi_values.iloc[-2]
            
            # RSI上穿超卖阈值（从下方穿越30）- 买入信号
            if prev_rsi < self.oversold and current_rsi >= self.oversold:
                return {'direction': 'buy', 'amount': 100}
            
            # RSI下穿超买阈值（从上方穿越70）- 卖出信号
            if prev_rsi > self.overbought and current_rsi <= self.overbought:
                return {'direction': 'sell', 'amount': 100}
        
        self.prev_rsi = current_rsi
        return None
