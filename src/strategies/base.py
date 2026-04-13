"""
策略基类模块
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd


class BaseStrategy(ABC):
    """
    策略基类
    
    所有自定义策略都需要继承此类并实现 on_bar 方法
    """
    
    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        self.params: Dict[str, Any] = {}
    
    def set_params(self, **kwargs):
        """设置策略参数"""
        self.params.update(kwargs)
    
    @abstractmethod
    def on_bar(self, context: Dict) -> Dict:
        """
        每个 Bar 调用一次，生成交易信号
        
        Args:
            context: 包含以下键的字典
                - date: 当前日期
                - price: 当前价格
                - portfolio: 当前投资组合
                - data: 历史数据（包含当前bar）
        
        Returns:
            交易信号字典，格式如下:
            - {'direction': 'buy', 'amount': 100}  # 买入100股
            - {'direction': 'sell', 'amount': 100} # 卖出100股
            - None  # 无信号
        """
        pass
    
    def on_start(self, context: Dict):
        """回测开始时调用"""
        pass
    
    def on_end(self, context: Dict):
        """回测结束时调用"""
        pass


class BuyAndHoldStrategy(BaseStrategy):
    """
    买入持有策略（示例）
    
    在第一天买入，持有到期末
    """
    
    def __init__(self):
        super().__init__("BuyAndHold")
        self.has_bought = False
    
    def on_bar(self, context: Dict) -> Dict:
        if not self.has_bought:
            self.has_bought = True
            return {'direction': 'buy', 'amount': 100}
        return None


class MovingAverageCrossStrategy(BaseStrategy):
    """
    移动平均线交叉策略（示例）
    
    短期均线上穿长期均线时买入，下穿时卖出
    """
    
    def __init__(self, short_window: int = 5, long_window: int = 20):
        super().__init__("MACross")
        self.short_window = short_window
        self.long_window = long_window
    
    def on_bar(self, context: Dict) -> Dict:
        data = context['data']
        
        if len(data) < self.long_window:
            return None
        
        short_ma = data['close'].tail(self.short_window).mean()
        long_ma = data['close'].tail(self.long_window).mean()
        
        # 获取前一天的均线
        if len(data) > self.long_window:
            prev_short_ma = data['close'].iloc[-self.short_window-1:-1].mean()
            prev_long_ma = data['close'].iloc[-self.long_window-1:-1].mean()
            
            # 金叉买入
            if prev_short_ma <= prev_long_ma and short_ma > long_ma:
                return {'direction': 'buy', 'amount': 100}
            
            # 死叉卖出
            if prev_short_ma >= prev_long_ma and short_ma < long_ma:
                return {'direction': 'sell', 'amount': 100}
        
        return None