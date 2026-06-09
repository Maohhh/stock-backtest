"""
农产品配对价差回归策略 (Pair Spread Reversion)

封装 indicators.spread_zscore，对协整农产品配对（玉米-淀粉等）做高胜率均值回归，
并产出**胜率、盈亏比、盈利因子**等交易统计与净值。

成本以"价差点数"计：每次开/平都要交易两条腿，cost_points 为单次进出的总成本（点）。
玉米/淀粉每点 = 10 元（10 吨/手 × 1 元/吨）。
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from ..indicators.spread_zscore import build_spread, spread_zscore, reversion_position


@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: int          # +1 多价差 / -1 空价差
    pnl_points: float       # 已扣成本（点）


@dataclass
class SpreadReversionResult:
    trades: List[Trade]
    point_value: float
    capital: float
    bars_index: pd.DatetimeIndex

    @property
    def pnl(self) -> np.ndarray:
        return np.array([t.pnl_points for t in self.trades], dtype=float)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        p = self.pnl
        return float((p > 0).mean()) if len(p) else 0.0

    @property
    def profit_factor(self) -> float:
        p = self.pnl
        gains, losses = p[p > 0].sum(), -p[p < 0].sum()
        return float(gains / losses) if losses > 0 else float("inf")

    @property
    def payoff_ratio(self) -> float:
        """平均盈利 / 平均亏损（盈亏比）。"""
        p = self.pnl
        w, l = p[p > 0], p[p < 0]
        if len(w) == 0 or len(l) == 0:
            return float("nan")
        return float(w.mean() / -l.mean())

    @property
    def annual_return(self) -> float:
        if not self.trades:
            return 0.0
        years = max((self.bars_index[-1] - self.bars_index[0]).days / 365.25, 1e-9)
        total_yuan = self.pnl.sum() * self.point_value
        return float(total_yuan / self.capital / years)

    def summary(self) -> dict:
        return {
            "交易笔数": self.n_trades,
            "胜率": round(self.win_rate * 100, 1),
            "盈亏比": round(self.payoff_ratio, 2),
            "盈利因子": round(self.profit_factor, 2),
            "年化收益(估)": round(self.annual_return * 100, 1),
            "总盈亏(点)": round(float(self.pnl.sum()), 1),
        }


@dataclass
class PairReversionStrategy:
    """
    协整配对价差回归策略。

    参数:
        window: Z 分数滚动窗口
        entry / exit / stop: 开仓 / 止盈 / 止损的 Z 阈值
        mode: 价差合成方式 'diff'（默认，同乘数品种）或 'ratio'
        beta: diff 模式对冲比例
        cost_points: 单次进出总成本（点，含两腿手续费+滑点）
        point_value: 每点价值（元），玉米/淀粉=10
        capital: 用于估算年化收益的占用资金（元，约两腿保证金）
    """
    window: int = 30
    entry: float = 2.0
    exit: float = 0.5
    stop: float = 4.0
    mode: str = "diff"
    beta: Optional[float] = None
    cost_points: float = 8.0
    point_value: float = 10.0
    capital: float = 5000.0

    def zscore(self, price_a: pd.Series, price_b: pd.Series) -> pd.Series:
        spread = build_spread(price_a, price_b, self.mode, self.beta)
        return spread_zscore(spread, self.window)

    def run(self, price_a: pd.Series, price_b: pd.Series) -> SpreadReversionResult:
        spread = build_spread(price_a, price_b, self.mode, self.beta)
        z = spread_zscore(spread, self.window)
        pos = reversion_position(z, self.entry, self.exit, self.stop)
        sp = spread.values
        trades: List[Trade] = []
        prev, entry_px, entry_dt, direction = 0, None, None, 0
        for i in range(len(pos)):
            cur = pos.iloc[i]
            if prev == 0 and cur != 0:                 # 开仓
                entry_px, entry_dt, direction = sp[i], spread.index[i], int(cur)
            elif prev != 0 and cur != prev:            # 平仓（含反手）
                pnl = (sp[i] - entry_px) * prev - self.cost_points
                trades.append(Trade(entry_dt, spread.index[i], int(prev), float(pnl)))
                if cur != 0:                           # 直接反手则立即再开
                    entry_px, entry_dt, direction = sp[i], spread.index[i], int(cur)
            prev = cur
        return SpreadReversionResult(trades, self.point_value, self.capital, spread.index)
