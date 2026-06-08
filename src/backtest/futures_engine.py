"""
期货组合（横截面）回测引擎

向量化回测一个多品种"目标权重面板"策略：
- 输入价格面板 + 目标权重面板（见 indicators.xs_reversal.target_weights）
- 计算逐 bar 组合收益、扣除换手成本、可选波动率目标杠杆
- 输出绩效指标与净值曲线

与单标的的 BacktestEngine 不同，这里处理的是横截面多空组合，
权重在 t 时刻确定、在 t+1 bar 实现收益（shift 1 避免前视）。
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class FuturesBacktestResult:
    """组合回测结果。"""
    equity: pd.Series            # 净值曲线（起点 1.0）
    returns: pd.Series           # 逐 bar 组合收益（已扣成本、已加杠杆）
    bars_per_year: float         # 年化用的每年 bar 数
    cost: float                  # 单边成本（小数）
    leverage: float              # 实际应用的杠杆倍数
    avg_turnover: float          # 平均每 bar 换手

    @property
    def ann_return(self) -> float:
        return float(self.returns.mean() * self.bars_per_year)

    @property
    def ann_vol(self) -> float:
        return float(self.returns.std() * np.sqrt(self.bars_per_year))

    @property
    def sharpe(self) -> float:
        sd = self.returns.std()
        return float(self.returns.mean() / sd * np.sqrt(self.bars_per_year)) if sd > 0 else 0.0

    @property
    def max_drawdown(self) -> float:
        return float((self.equity / self.equity.cummax() - 1).min())

    @property
    def total_return(self) -> float:
        return float(self.equity.iloc[-1] - 1)

    def summary(self) -> dict:
        return {
            "年化收益": round(self.ann_return * 100, 1),
            "夏普比率": round(self.sharpe, 2),
            "最大回撤": round(self.max_drawdown * 100, 1),
            "年化波动": round(self.ann_vol * 100, 1),
            "累计收益": round(self.total_return * 100, 1),
            "杠杆": round(self.leverage, 2),
            "平均换手/bar": round(self.avg_turnover, 3),
            "bar数": len(self.returns),
        }


def bars_per_year(prices: pd.DataFrame) -> float:
    """根据面板估算每年 bar 数（每日 bar 数 × 252）。"""
    days = len(pd.Index(prices.index).normalize().unique())
    return 252.0 * len(prices) / max(days, 1)


def run_panel_backtest(prices: pd.DataFrame, weights: pd.DataFrame,
                       cost: float = 0.0002, target_vol: Optional[float] = None,
                       max_leverage: float = 4.0) -> FuturesBacktestResult:
    """
    回测一个目标权重面板策略。

    参数:
        prices: 价格面板（列=品种）
        weights: 目标权重面板（与 prices 对齐）
        cost: 单边交易成本（小数，0.0002=2bp，含手续费+滑点）
        target_vol: 若给定，则按事后整体波动率缩放到该年化目标（杠杆=target/realized）
        max_leverage: 杠杆上限，防止 target_vol 过度放大

    返回:
        FuturesBacktestResult
    """
    prices, weights = prices.align(weights, join="inner")
    ret = prices.pct_change()
    pos = weights.shift(1).fillna(0.0)                 # t 时刻持仓，t+1 实现
    gross = (pos * ret).sum(axis=1)
    turnover = pos.diff().abs().sum(axis=1)
    net = (gross - turnover * cost).dropna()

    bpy = bars_per_year(prices)
    leverage = 1.0
    if target_vol is not None:
        realized = net.std() * np.sqrt(bpy)
        if realized > 0:
            leverage = min(target_vol / realized, max_leverage)
            net = net * leverage

    equity = (1 + net).cumprod()
    return FuturesBacktestResult(
        equity=equity, returns=net, bars_per_year=bpy, cost=cost,
        leverage=leverage, avg_turnover=float(turnover.mean()),
    )
