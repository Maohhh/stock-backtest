"""
回测引擎模块
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Order:
    """订单"""
    symbol: str
    direction: str  # 'buy' or 'sell'
    amount: int
    price: float
    timestamp: datetime


@dataclass
class Position:
    """持仓"""
    symbol: str
    amount: int
    avg_cost: float
    current_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        return self.amount * self.current_price
    
    @property
    def pnl(self) -> float:
        return self.amount * (self.current_price - self.avg_cost)


@dataclass
class Portfolio:
    """投资组合"""
    cash: float
    positions: Dict[str, Position]
    
    @property
    def total_value(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions.values())
    
    @property
    def total_pnl(self) -> float:
        return sum(p.pnl for p in self.positions.values())


class BacktestEngine:
    """
    回测引擎
    
    支持日级回测，记录交易历史，计算绩效指标
    """
    
    def __init__(self, initial_cash: float = 100000.0, commission: float = 0.0003):
        """
        初始化回测引擎
        
        Args:
            initial_cash: 初始资金
            commission: 手续费率（默认万分之三）
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.portfolio = Portfolio(cash=initial_cash, positions={})
        self.orders: List[Order] = []
        self.trades: List[Dict] = []
        self.daily_values: List[Dict] = []
        self.current_date: Optional[datetime] = None
    
    def run(self, strategy, data: pd.DataFrame) -> Dict:
        """
        运行回测
        
        Args:
            strategy: 策略实例
            data: 价格数据 DataFrame
            
        Returns:
            回测结果字典
        """
        print(f"🚀 开始回测，初始资金: {self.initial_cash:,.2f}")
        
        for idx, row in data.iterrows():
            self.current_date = row['date']
            
            # 更新持仓价格
            symbol = data.attrs.get('symbol', 'UNKNOWN')
            if symbol in self.portfolio.positions:
                self.portfolio.positions[symbol].current_price = row['close']
            
            # 调用策略生成信号
            context = {
                'date': self.current_date,
                'price': row['close'],
                'portfolio': self.portfolio,
                'data': data.loc[:idx]
            }
            
            signal = strategy.on_bar(context)
            
            if signal:
                self._execute_signal(symbol, signal, row['close'])
            
            # 记录每日净值
            self.daily_values.append({
                'date': self.current_date,
                'total_value': self.portfolio.total_value,
                'cash': self.portfolio.cash,
                'positions_value': self.portfolio.total_value - self.portfolio.cash
            })
        
        return self._generate_report()
    
    def _execute_signal(self, symbol: str, signal: Dict, price: float):
        """执行交易信号"""
        direction = signal.get('direction')
        amount = signal.get('amount', 0)
        
        if direction == 'buy':
            cost = amount * price * (1 + self.commission)
            if cost <= self.portfolio.cash:
                self.portfolio.cash -= cost
                
                if symbol in self.portfolio.positions:
                    pos = self.portfolio.positions[symbol]
                    total_cost = pos.amount * pos.avg_cost + amount * price
                    pos.amount += amount
                    pos.avg_cost = total_cost / pos.amount
                else:
                    self.portfolio.positions[symbol] = Position(
                        symbol=symbol,
                        amount=amount,
                        avg_cost=price,
                        current_price=price
                    )
                
                self.trades.append({
                    'date': self.current_date,
                    'symbol': symbol,
                    'direction': 'buy',
                    'amount': amount,
                    'price': price,
                    'cost': cost
                })
        
        elif direction == 'sell':
            if symbol in self.portfolio.positions:
                pos = self.portfolio.positions[symbol]
                sell_amount = min(amount, pos.amount)
                revenue = sell_amount * price * (1 - self.commission)
                
                self.portfolio.cash += revenue
                pos.amount -= sell_amount
                
                if pos.amount == 0:
                    del self.portfolio.positions[symbol]
                
                self.trades.append({
                    'date': self.current_date,
                    'symbol': symbol,
                    'direction': 'sell',
                    'amount': sell_amount,
                    'price': price,
                    'revenue': revenue
                })
    
    def _generate_report(self) -> Dict:
        """生成回测报告"""
        if not self.daily_values:
            return {}
        
        values_df = pd.DataFrame(self.daily_values)
        
        # 计算收益率
        total_return = (self.portfolio.total_value - self.initial_cash) / self.initial_cash
        
        # 计算最大回撤
        values_df['cummax'] = values_df['total_value'].cummax()
        values_df['drawdown'] = (values_df['total_value'] - values_df['cummax']) / values_df['cummax']
        max_drawdown = values_df['drawdown'].min()
        
        # 计算夏普比率（简化版，假设无风险利率为0）
        returns = values_df['total_value'].pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        
        report = {
            'initial_cash': self.initial_cash,
            'final_value': self.portfolio.total_value,
            'total_return': total_return,
            'total_return_pct': f"{total_return * 100:.2f}%",
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': f"{max_drawdown * 100:.2f}%",
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(self.trades),
            'daily_values': values_df,
            'trades': pd.DataFrame(self.trades)
        }
        
        print(f"\n📊 回测结果:")
        print(f"   初始资金: {self.initial_cash:,.2f}")
        print(f"   最终资产: {self.portfolio.total_value:,.2f}")
        print(f"   总收益率: {report['total_return_pct']}")
        print(f"   最大回撤: {report['max_drawdown_pct']}")
        print(f"   夏普比率: {sharpe_ratio:.2f}")
        print(f"   交易次数: {len(self.trades)}")
        
        return report