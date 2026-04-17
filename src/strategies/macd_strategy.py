"""
MACD 策略 (Moving Average Convergence Divergence Strategy)

基于MACD金叉死叉的交易策略：
- DIF上穿DEA（金叉）时买入
- DIF下穿DEA（死叉）时卖出
"""

from typing import Dict
import pandas as pd
from .base import BaseStrategy
from ..indicators.macd import macd


class MACDStrategy(BaseStrategy):
    """
    MACD指数平滑异同移动平均线策略
    
    策略逻辑：
    - 当DIF线从下向上穿越DEA线（金叉）时，产生买入信号
    - 当DIF线从上向下穿越DEA线（死叉）时，产生卖出信号
    
    参数:
        fast: 快速EMA周期，默认12
        slow: 慢速EMA周期，默认26
        signal: 信号线EMA周期，默认9
    """
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__("MACDStrategy")
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def set_params(self, **kwargs):
        """设置策略参数"""
        super().set_params(**kwargs)
        if 'fast' in kwargs:
            self.fast = kwargs['fast']
        if 'slow' in kwargs:
            self.slow = kwargs['slow']
        if 'signal' in kwargs:
            self.signal = kwargs['signal']
    
    def on_bar(self, context: Dict) -> Dict:
        """
        每个Bar调用一次，生成交易信号
        
        Args:
            context: 包含date, price, portfolio, data的字典
        
        Returns:
            交易信号字典或None
        """
        data = context['data']
        
        # 数据不足时无法计算MACD
        min_periods = max(self.fast, self.slow) + self.signal
        if len(data) < min_periods:
            return None
        
        # 计算MACD
        macd_df = macd(data, fast=self.fast, slow=self.slow, signal=self.signal)
        
        # 获取当前和前一天的DIF和DEA
        current_dif = macd_df['DIF'].iloc[-1]
        current_dea = macd_df['DEA'].iloc[-1]
        
        if len(macd_df) > 1:
            prev_dif = macd_df['DIF'].iloc[-2]
            prev_dea = macd_df['DEA'].iloc[-2]
            
            # 金叉：DIF上穿DEA - 买入信号
            if prev_dif <= prev_dea and current_dif > current_dea:
                return {'direction': 'buy', 'amount': 100}
            
            # 死叉：DIF下穿DEA - 卖出信号
            if prev_dif >= prev_dea and current_dif < current_dea:
                return {'direction': 'sell', 'amount': 100}
        
        return None
