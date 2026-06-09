"""
农产品配对价差回归 指标/策略 单元测试
"""

import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indicators.spread_zscore import build_spread, spread_zscore, reversion_position
from src.strategies.spread_reversion import PairReversionStrategy


def _cointegrated_pair(n=600, seed=1):
    """构造一对协整序列：共同随机游走 + 均值回归的价差。"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    common = 2000 + np.cumsum(rng.normal(0, 5, n))
    spread = np.zeros(n)                       # AR(1) 均值回归价差
    for i in range(1, n):
        spread[i] = 0.9 * spread[i - 1] + rng.normal(0, 8)
    a = pd.Series(common, index=idx)
    b = pd.Series(common + 300 + spread, index=idx)
    return a, b


class TestSpreadReversion(unittest.TestCase):

    def setUp(self):
        self.a, self.b = _cointegrated_pair()

    def test_zscore_centered(self):
        z = spread_zscore(build_spread(self.a, self.b), 30).dropna()
        self.assertAlmostEqual(z.mean(), 0, delta=0.3)
        self.assertAlmostEqual(z.std(), 1, delta=0.3)

    def test_position_values(self):
        z = spread_zscore(build_spread(self.a, self.b), 30)
        pos = reversion_position(z, 1.5, 0.3, 4.0)
        self.assertTrue(set(pos.unique()).issubset({-1.0, 0.0, 1.0}))

    def test_position_opens_on_extreme(self):
        # 人为造一个极端 Z 应触发开仓
        z = pd.Series([np.nan, 0.0, 2.0, 0.1, -2.0, 0.0],
                      index=pd.date_range("2020", periods=6))
        pos = reversion_position(z, 1.5, 0.3, 4.0)
        self.assertEqual(pos.iloc[2], -1)   # z=2 > entry -> 空价差
        self.assertEqual(pos.iloc[3], 0)    # z=0.1 <= exit -> 平
        self.assertEqual(pos.iloc[4], 1)    # z=-2 < -entry -> 多价差

    def test_high_win_rate_on_mean_reverting(self):
        res = PairReversionStrategy(window=30, cost_points=0.0).run(self.a, self.b)
        self.assertGreater(res.n_trades, 10)
        self.assertGreater(res.win_rate, 0.55)         # 均值回归应高胜率
        self.assertGreater(res.profit_factor, 1.0)

    def test_metrics_finite(self):
        res = PairReversionStrategy().run(self.a, self.b)
        for k, v in res.summary().items():
            self.assertTrue(np.isfinite(v), f"{k} 非有限值")

    def test_cost_reduces_pnl(self):
        no_cost = PairReversionStrategy(cost_points=0.0).run(self.a, self.b).pnl.sum()
        with_cost = PairReversionStrategy(cost_points=20.0).run(self.a, self.b).pnl.sum()
        self.assertGreater(no_cost, with_cost)


if __name__ == "__main__":
    unittest.main()
