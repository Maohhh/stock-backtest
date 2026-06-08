"""
单标的趋势跟踪策略 (Trend Following, single-instrument)

把 indicators.trend_ensemble 的趋势信号，叠加**波动率目标仓位管理**（按近期波动反比
调整手数，让风险恒定），构成一个完整的单品种方向性策略。复用组合回测引擎计算绩效。

"不爆仓"的三道保险：
1. ER 过滤——震荡不交易（约 70% 时间空仓）；
2. 波动率目标——波动放大时自动减仓，杠杆有上限 max_leverage；
3. 趋势跟踪天然正偏度——亏损单被趋势反转及时止损、盈利单让利润奔跑。
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from ..indicators.trend_ensemble import trend_signal, DEFAULT_PAIRS
from ..backtest.futures_engine import run_panel_backtest, FuturesBacktestResult


@dataclass
class TrendFollowingStrategy:
    """
    单标的趋势跟踪。

    参数:
        pairs: 多周期 EMA 周期对
        er_n / er_th: 效率系数周期与阈值（震荡过滤）
        target_vol: 目标年化波动（用它决定杠杆；越大年化越高、回撤也越大）
        vol_lookback: 估计近期波动的回看天数
        max_leverage: 单标的杠杆上限（防止低波动时过度放大）
        cost: 单边交易成本（小数，按你的手续费+滑点）
        bars_per_year: 年化基数（日线=252，周线=52，60分钟≈252*4 ...）
    """
    pairs: List[Tuple[int, int]] = field(default_factory=lambda: list(DEFAULT_PAIRS))
    er_n: int = 20
    er_th: float = 0.3
    target_vol: float = 0.30
    vol_lookback: int = 20
    max_leverage: float = 3.0
    cost: float = 0.0002
    bars_per_year: int = 252

    def position(self, df: pd.DataFrame) -> pd.Series:
        """目标仓位（已含波动率目标缩放），范围约 [-max_leverage, max_leverage]。"""
        sig = trend_signal(df, self.pairs, self.er_n, self.er_th)
        ret = df["close"].pct_change()
        realized = ret.rolling(self.vol_lookback).std() * np.sqrt(self.bars_per_year)
        scale = (self.target_vol / realized).clip(0, self.max_leverage)
        return (sig * scale).shift(0)   # 当日信号，回测里再 shift(1) 避免前视

    def run(self, df: pd.DataFrame, symbol: str = "X") -> FuturesBacktestResult:
        prices = df[["close"]].rename(columns={"close": symbol})
        weights = self.position(df).to_frame(symbol)
        res = run_panel_backtest(prices, weights, cost=self.cost, target_vol=None)
        # 用本策略设定的年化基数覆盖（引擎默认按日频估算）
        res.bars_per_year = self.bars_per_year
        return res
