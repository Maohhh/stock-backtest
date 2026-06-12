"""
价差套利回测模块

针对跨期价差（如菜粕 9-11）和跨品种价差（如 PTA-短纤）的均值回归套利策略，
支持「低于阈值做多 / 高于阈值做空 -> 回归到中枢平仓」的回测，并在到达指定
日期（跨期价差的交割月前）强制平仓。

约定：
- 价差 S = 近月腿收盘 - 远月腿收盘（或 品种A - 品种B），单位为价差点数（元/吨，
  鸡蛋为元/500千克）。
- 做多盈亏 = 出场价差 - 入场价差；做空盈亏 = 入场价差 - 出场价差。
- 同一时间至多持有一个方向的仓位（不加仓）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Trade:
    direction: str          # 'long' / 'short'
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_spread: float
    exit_spread: float
    target: float
    exit_reason: str        # 'revert' / 'force_close'
    points: float           # 价差点数盈亏
    mae: float              # 最大不利波动（点数）
    hold_days: int


@dataclass
class SpreadResult:
    name: str
    trades: list[Trade] = field(default_factory=list)
    equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    multiplier: float = 1.0  # 价差 1 点对应的每手对盈亏（元）

    # ---- 汇总统计 ----
    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def total_points(self) -> float:
        return float(sum(t.points for t in self.trades))

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return float("nan")
        wins = sum(1 for t in self.trades if t.points > 0)
        return wins / len(self.trades)

    @property
    def avg_points(self) -> float:
        return self.total_points / self.n_trades if self.trades else float("nan")

    @property
    def avg_hold_days(self) -> float:
        if not self.trades:
            return float("nan")
        return float(np.mean([t.hold_days for t in self.trades]))

    @property
    def best(self) -> float:
        return max((t.points for t in self.trades), default=float("nan"))

    @property
    def worst(self) -> float:
        return min((t.points for t in self.trades), default=float("nan"))

    @property
    def total_yuan(self) -> float:
        """近似每手对累计盈亏（元，毛利，未计手续费/滑点）。"""
        return self.total_points * self.multiplier

    @property
    def max_drawdown_points(self) -> float:
        """累计点数权益曲线的最大回撤（点数）。"""
        if self.equity.empty:
            return 0.0
        cummax = self.equity.cummax()
        return float((self.equity - cummax).min())

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([t.__dict__ for t in self.trades])


def backtest_spread(
    spread: pd.Series,
    name: str,
    *,
    entry_long: Optional[float] = None,
    entry_short: Optional[float] = None,
    exit_mode: str = "zero",          # 'zero' / 'cycle_mean' / 'rolling_mean'
    rolling_window: int = 120,
    min_history: int = 20,
    force_close: bool = False,
    slippage: float = 0.0,
    multiplier: float = 1.0,
) -> SpreadResult:
    """对单条价差序列执行均值回归套利回测。

    Parameters
    ----------
    spread : pd.Series
        以日期为索引、按日期升序的价差序列（已剔除交割月）。
    entry_long / entry_short : float | None
        做多 / 做空的进场阈值。``S < entry_long`` 做多；``S > entry_short`` 做空。
    exit_mode : str
        平仓中枢的确定方式：
        - ``zero``        ：中枢固定为 0（围绕 0 波动的价差，如花生 3-4）。
        - ``cycle_mean``  ：进场时本周期的扩张均值（买压缩价差，回归到均值）。
        - ``rolling_mean``：进场时的滚动均值（跨品种主力连续价差的中枢）。
    rolling_window : int
        ``rolling_mean`` 模式的窗口（交易日）。
    min_history : int
        允许进场前所需的最少历史样本数（避免周期初期均值不稳）。
    force_close : bool
        序列末根 K 线是否强制平仓（跨期价差交割月前强平）。
    slippage : float
        每笔交易（一进一出）扣除的价差点数，模拟双边滑点 / 手续费。
    multiplier : float
        价差 1 点对应的每手对盈亏（元），用于折算名义盈亏。
    """
    s = spread.dropna().sort_index()
    if s.empty:
        return SpreadResult(name=name, multiplier=multiplier)

    if exit_mode == "rolling_mean":
        center = s.rolling(rolling_window, min_periods=min_history).mean()
    elif exit_mode == "cycle_mean":
        center = s.expanding(min_periods=min_history).mean()
    else:  # zero
        center = pd.Series(0.0, index=s.index)

    pos = 0                # 0 flat / +1 long / -1 short
    entry_idx = entry_spread = target = mae = None
    trades: list[Trade] = []
    realized = 0.0
    equity_dates: list[pd.Timestamp] = []
    equity_vals: list[float] = []

    dates = s.index
    n = len(dates)
    for i, dt in enumerate(dates):
        val = s.iloc[i]
        c = center.iloc[i]
        is_last = i == n - 1

        # 1) 更新最大不利波动
        if pos != 0:
            if pos > 0:
                mae = min(mae, val - entry_spread)
            else:
                mae = min(mae, entry_spread - val)

        # 2) 平仓判断
        if pos != 0:
            hit = (pos > 0 and val >= target) or (pos < 0 and val <= target)
            reason = None
            if hit:
                reason = "revert"
            elif force_close and is_last:
                reason = "force_close"
            if reason:
                pts = (val - entry_spread) if pos > 0 else (entry_spread - val)
                pts -= slippage
                realized += pts
                trades.append(
                    Trade(
                        direction="long" if pos > 0 else "short",
                        entry_date=entry_idx,
                        exit_date=dt,
                        entry_spread=float(entry_spread),
                        exit_spread=float(val),
                        target=float(target),
                        exit_reason=reason,
                        points=float(pts),
                        mae=float(mae),
                        hold_days=int((dt - entry_idx).days),
                    )
                )
                pos = 0

        # 3) 开仓判断（平仓后同根不再反向开仓；末根不再开仓）
        if pos == 0 and not is_last and not np.isnan(c):
            if entry_long is not None and val < entry_long and val < c:
                pos, entry_idx, entry_spread, target, mae = 1, dt, val, c, 0.0
            elif entry_short is not None and val > entry_short and val > c:
                pos, entry_idx, entry_spread, target, mae = -1, dt, val, c, 0.0

        # 4) 记录权益曲线（含浮动盈亏）
        floating = 0.0
        if pos > 0:
            floating = val - entry_spread
        elif pos < 0:
            floating = entry_spread - val
        equity_dates.append(dt)
        equity_vals.append(realized + floating)

    result = SpreadResult(name=name, trades=trades, multiplier=multiplier)
    result.equity = pd.Series(equity_vals, index=pd.DatetimeIndex(equity_dates), name=name)
    return result
