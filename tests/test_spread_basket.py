"""
跨品种套利篮子 / 价差段回测 单元测试
"""

import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategies.spread_basket import backtest_segments, SpreadBasket, _trade_stats


def _revert_segment(n=200, seed=0, drift=0.0):
    """构造一段均值回归价差。"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + rng.normal(0, 5) + drift
    return pd.Series(x + 100, index=idx)


class TestSpreadBasket(unittest.TestCase):

    def test_trade_stats_empty(self):
        st = _trade_stats(np.array([]))
        self.assertEqual(st["n_trades"], 0)

    def test_backtest_segments_metrics(self):
        segs = [_revert_segment(seed=i) for i in range(4)]
        series, st = backtest_segments(segs, cost=0.0)
        self.assertGreater(st["n_trades"], 5)
        self.assertGreater(st["win_rate"], 0.5)        # 回归段应高胜率
        self.assertGreater(st["profit_factor"], 1.0)
        self.assertIsInstance(series, pd.Series)

    def test_cost_reduces_total(self):
        segs = [_revert_segment(seed=i) for i in range(4)]
        free = backtest_segments(segs, cost=0.0)[1]["total"]
        paid = backtest_segments(segs, cost=10.0)[1]["total"]
        self.assertGreater(free, paid)

    def test_basket_diversifies(self):
        # 两条相互独立的回归腿，组合波动应被分散（夏普不为 0）
        legs = {
            "A": [_revert_segment(seed=i) for i in range(6)],
            "B": [_revert_segment(seed=100 + i) for i in range(6)],
        }
        res = SpreadBasket(legs, cost=0.0).run()
        self.assertIn("combo_ann_sharpe", res)
        self.assertEqual(set(res["leg_ann_sharpe"]), {"A", "B"})
        self.assertTrue(np.isfinite(res["combo_ann_sharpe"]))
        self.assertGreaterEqual(res["combo_win_month"], 0.0)


if __name__ == "__main__":
    unittest.main()
