"""
布林带策略 (Bollinger Bands Strategy)

基于布林带指标的交易策略：
- 价格触及下轨时买入
- 价格触及上轨时卖出
"""

from typing import Dict
import pandas as pd
from .base import BaseStrategy
from ..indicators.bollinger import bollinger_bands


class BollingerStrategy(BaseStrategy):
    """
    布林带交易策略
    
    策略逻辑：
    - 当价格从下向上穿越布林带下轨时，产生买入信号
    - 当价格从上向下穿越布林带上轨时，产生卖出信号
    
    参数:
        period: 移动平均周期，默认20
        std_dev: 标准差倍数，默认2.0
    """
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__("BollingerStrategy")
        self.period = period
        self.std_dev = std_dev
    
    def set_params(self, **kwargs):
        """设置策略参数"""
        super().set_params(**kwargs)
        if 'period' in kwargs:
            self.period = kwargs['period']
        if 'std_dev' in kwargs:
            self.std_dev = kwargs['std_dev']
    
    def on_bar(self, context: Dict) -> Dict:
        """
        每个Bar调用一次，生成交易信号
        
        Args:
            context: 包含date, price, portfolio, data的字典
        
        Returns:
            交易信号字典或None
        """
        data = context['data']
        
        # 数据不足时无法计算布林带
        if len(data) < self.period + 1:
            return None
        
        # 计算布林带
        bb = bollinger_bands(data, period=self.period, std_dev=self.std_dev)
        
        # 获取当前和前一天的收盘价及布林带值
        current_close = data['close'].iloc[-1]
        current_lower = bb['lower'].iloc[-1]
        current_upper = bb['upper'].iloc[-1]
        
        if len(data) > 1:
            prev_close = data['close'].iloc[-2]
            prev_lower = bb['lower'].iloc[-2]
            prev_upper = bb['upper'].iloc[-2]
            
            # 价格从下向上穿越下轨 - 买入信号
            if prev_close <= prev_lower and current_close > current_lower:
                return {'direction': 'buy', 'amount': 100}
            
            # 价格从上向下穿越上轨 - 卖出信号
            if prev_close >= prev_upper and current_close < current_upper:
                return {'direction': 'sell', 'amount': 100}
        
        return None
