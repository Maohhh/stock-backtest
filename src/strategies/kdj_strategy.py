"""
KDJ 策略 (Stochastic Oscillator Strategy)

基于KDJ随机指标的交易策略：
- K线上穿D线且J值<20时买入
- K线下穿D线且J值>80时卖出
"""

from typing import Dict
import pandas as pd
from .base import BaseStrategy
from ..indicators.kdj import kdj


class KDJStrategy(BaseStrategy):
    """
    KDJ随机指标策略
    
    策略逻辑：
    - 当K线上穿D线（金叉）且J值<20（超卖区）时，产生买入信号
    - 当K线下穿D线（死叉）且J值>80（超买区）时，产生卖出信号
    
    参数:
        n: RSV计算周期，默认9
        m1: K线平滑周期，默认3
        m2: D线平滑周期，默认3
        j_buy_threshold: J值买入阈值，默认20
        j_sell_threshold: J值卖出阈值，默认80
    """
    
    def __init__(self, n: int = 9, m1: int = 3, m2: int = 3, 
                 j_buy_threshold: float = 20, j_sell_threshold: float = 80):
        super().__init__("KDJStrategy")
        self.n = n
        self.m1 = m1
        self.m2 = m2
        self.j_buy_threshold = j_buy_threshold
        self.j_sell_threshold = j_sell_threshold
    
    def set_params(self, **kwargs):
        """设置策略参数"""
        super().set_params(**kwargs)
        if 'n' in kwargs:
            self.n = kwargs['n']
        if 'm1' in kwargs:
            self.m1 = kwargs['m1']
        if 'm2' in kwargs:
            self.m2 = kwargs['m2']
        if 'j_buy_threshold' in kwargs:
            self.j_buy_threshold = kwargs['j_buy_threshold']
        if 'j_sell_threshold' in kwargs:
            self.j_sell_threshold = kwargs['j_sell_threshold']
    
    def on_bar(self, context: Dict) -> Dict:
        """
        每个Bar调用一次，生成交易信号
        
        Args:
            context: 包含date, price, portfolio, data的字典
        
        Returns:
            交易信号字典或None
        """
        data = context['data']
        
        # 数据不足时无法计算KDJ
        min_periods = self.n + max(self.m1, self.m2)
        if len(data) < min_periods:
            return None
        
        # 检查必需的列
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                return None
        
        # 计算KDJ
        kdj_df = kdj(data, n=self.n, m1=self.m1, m2=self.m2)
        
        # 获取当前值
        current_k = kdj_df['K'].iloc[-1]
        current_d = kdj_df['D'].iloc[-1]
        current_j = kdj_df['J'].iloc[-1]
        
        if len(kdj_df) > 1:
            prev_k = kdj_df['K'].iloc[-2]
            prev_d = kdj_df['D'].iloc[-2]
            
            # 金叉：K线上穿D线且J值在超卖区 - 买入信号
            if prev_k <= prev_d and current_k > current_d and current_j < self.j_buy_threshold:
                return {'direction': 'buy', 'amount': 100}
            
            # 死叉：K线下穿D线且J值在超买区 - 卖出信号
            if prev_k >= prev_d and current_k < current_d and current_j > self.j_sell_threshold:
                return {'direction': 'sell', 'amount': 100}
        
        return None
