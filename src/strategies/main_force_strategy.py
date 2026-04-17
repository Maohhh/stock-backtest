"""
主力共振策略 (Main Force Resonance Strategy)

基于主力资金流向和买卖共振信号的复合策略：
- 强买点：XPCT极低(<=0)且主力净流入为正
- 弱买点：XPCT极低(<=0)
- 卖出信号：卖共振(XPCT>=95且出货)且资金流出
"""

import pandas as pd
import numpy as np
from .base import BaseStrategy
from ..indicators import max_force_resonance


class MainForceResonanceStrategy(BaseStrategy):
    """
    主力共振策略
    
    买入条件：
    - 强买点: XPCT <= 0 且 主力净流入 > 0
    - 或弱买点: XPCT <= 0 (作为备选)
    
    卖出条件：
    - XG100_S信号: XPCT >= 95 且 出货 且 (TMP < 0 或 主力净流入 < 0)
    
    参数:
        n: XPCT计算周期，默认12
        m: XPCT高低点周期，默认240
        bp_buy: 买入阈值，默认0
        sp_sell: 卖出阈值，默认95
        use_strong_buy_only: 是否只使用强买点，默认True
    """
    
    def __init__(self, n: int = 12, m: int = 240, 
                 bp_buy: float = 0, sp_sell: float = 95,
                 use_strong_buy_only: bool = True):
        super().__init__("主力共振策略")
        self.n = n
        self.m = m
        self.bp_buy = bp_buy
        self.sp_sell = sp_sell
        self.use_strong_buy_only = use_strong_buy_only
        self.params = {
            'n': n,
            'm': m,
            'bp_buy': bp_buy,
            'sp_sell': sp_sell,
            'use_strong_buy_only': use_strong_buy_only
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        返回DataFrame包含：
        - signal: 1(买入), -1(卖出), 0(持仓)
        - position: 建议持仓比例
        - xpct: XPCT指标值
        - main_net_inflow: 主力净流入
        - signal_type: 信号类型描述
        """
        # 计算主力共振指标
        mfr = max_force_resonance(df, self.n, self.m, self.bp_buy, self.sp_sell)
        
        result = pd.DataFrame(index=df.index)
        result['xpct'] = mfr['xpct']
        result['main_net_inflow'] = mfr['main_net_inflow']
        result['main_force_entry'] = mfr['main_force_entry']
        result['ship'] = mfr['ship']
        
        # 生成信号
        result['signal'] = 0
        result['signal_type'] = ''
        
        # 买入信号
        if self.use_strong_buy_only:
            # 只使用强买点
            buy_condition = mfr['strong_buy']
            result.loc[buy_condition, 'signal_type'] = '强买点'
        else:
            # 使用强买点和弱买点
            strong_buy = mfr['strong_buy']
            weak_buy = mfr['weak_buy'] & ~mfr['strong_buy']
            buy_condition = strong_buy | weak_buy
            result.loc[strong_buy, 'signal_type'] = '强买点'
            result.loc[weak_buy, 'signal_type'] = '弱买点'
        
        result.loc[buy_condition, 'signal'] = 1
        
        # 卖出信号
        sell_condition = mfr['xg100_s']
        result.loc[sell_condition, 'signal'] = -1
        result.loc[sell_condition, 'signal_type'] = '卖出信号'
        
        # 计算建议持仓比例
        result['position'] = 0.0
        
        # 基于XPCT计算持仓比例
        # XPCT越低，持仓比例越高（反向关系）
        xpct_norm = 1 - (result['xpct'] / 100)  # 0-1范围，XPCT高则值低
        
        # 只在买入信号时建立仓位
        for i in range(len(result)):
            if result['signal'].iloc[i] == 1:
                # 强买点全仓，弱买点半仓
                if result['signal_type'].iloc[i] == '强买点':
                    result.loc[result.index[i], 'position'] = 1.0
                elif result['signal_type'].iloc[i] == '弱买点':
                    result.loc[result.index[i], 'position'] = 0.5
            elif result['signal'].iloc[i] == -1:
                result.loc[result.index[i], 'position'] = 0.0
            elif i > 0:
                # 无信号时保持上一期持仓
                result.loc[result.index[i], 'position'] = result['position'].iloc[i-1]
        
        return result
    
    def on_bar(self, context: dict) -> dict:
        """
        每个Bar调用一次，生成交易信号
        
        Args:
            context: 包含当前bar数据的字典
            
        Returns:
            交易信号字典，包含action和amount
        """
        # 获取当前数据
        df = context.get('data', pd.DataFrame())
        if df.empty:
            return {'action': 'hold', 'amount': 0}
        
        # 生成信号
        signals = self.generate_signals(df)
        
        # 获取最新信号
        if signals.empty:
            return {'action': 'hold', 'amount': 0}
        
        latest = signals.iloc[-1]
        signal = latest['signal']
        position = latest['position']
        
        # 转换为交易动作
        if signal == 1:
            return {'action': 'buy', 'amount': position}
        elif signal == -1:
            return {'action': 'sell', 'amount': 1.0}
        else:
            return {'action': 'hold', 'amount': 0}
    
    def get_required_columns(self) -> list:
        """返回策略所需的数据列"""
        return ['open', 'high', 'low', 'close', 'volume']
    
    def get_indicator_config(self) -> dict:
        """返回指标配置参数"""
        return {
            'name': '主力共振指标',
            'params': self.params,
            'description': '基于主力资金流向和XPCT指标的复合策略'
        }
