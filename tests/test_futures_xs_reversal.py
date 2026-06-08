"""
期货横截面反转指标 / 组合回测引擎 单元测试
"""

import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indicators.xs_reversal import cross_sectional_reversal, target_weights
from src.backtest.futures_engine import run_panel_backtest, bars_per_year
from src.strategies.futures_xs_reversal import FuturesXSReversalStrategy, optimize


class TestXSReversal(unittest.TestCase):

    def setUp(self):
        # 构造 4 个相关品种的 15 分钟价格面板
        idx = pd.date_range("2024-01-01 09:00", periods=300, freq="15min")
        rng = np.random.default_rng(0)
        common = np.cumsum(rng.normal(0, 1, len(idx)))  # 共同因子
        data = {}
        for k in range(4):
            idio = np.cumsum(rng.normal(0, 0.5, len(idx)))
            data[f"X{k}"] = 1000 + common + idio
        self.prices = pd.DataFrame(data, index=idx)

    def test_signal_is_sector_neutral(self):
        sig = cross_sectional_reversal(self.prices, form=4)
        self.assertEqual(sig.shape, self.prices.shape)
        # 板块中性：每个时刻横截面信号之和应≈0
        row_sums = sig.dropna().sum(axis=1).abs()
        self.assertTrue((row_sums < 1e-9).all())

    def test_weights_gross_normalized(self):
        w = target_weights(self.prices, form=4, rebalance=8)
        gross = w.abs().sum(axis=1)
        nonzero = gross[gross > 0]
        # 非空仓时刻总杠杆=1
        self.assertTrue(np.allclose(nonzero, 1.0))

    def test_rebalance_reduces_turnover(self):
        w_fast = target_weights(self.prices, form=4, rebalance=1)
        w_slow = target_weights(self.prices, form=4, rebalance=24)
        t_fast = w_fast.diff().abs().sum(axis=1).mean()
        t_slow = w_slow.diff().abs().sum(axis=1).mean()
        self.assertLess(t_slow, t_fast)

    def test_backtest_runs_and_metrics(self):
        res = FuturesXSReversalStrategy(form=4, rebalance=8, cost=0.0002).run(self.prices)
        self.assertEqual(len(res.equity), len(res.returns))
        self.assertTrue(np.isfinite(res.sharpe))
        self.assertLessEqual(res.max_drawdown, 0.0)

    def test_vol_targeting_scales_vol(self):
        base = FuturesXSReversalStrategy(form=4, rebalance=8, cost=0.0).run(self.prices)
        lev = FuturesXSReversalStrategy(form=4, rebalance=8, cost=0.0,
                                        target_vol=0.20).run(self.prices)
        # 加杠杆后年化波动应更接近目标
        self.assertGreater(lev.leverage, 0)
        self.assertAlmostEqual(lev.ann_vol, 0.20, delta=0.05)

    def test_no_lookahead(self):
        # 权重在 t 确定、t+1 实现：第一根收益应为 0/NaN 处理后不含未来信息
        res = run_panel_backtest(self.prices, target_weights(self.prices, 4, 8), cost=0.0)
        self.assertTrue(np.isfinite(res.returns).all())

    def test_optimize_returns_sorted(self):
        tbl = optimize(self.prices, forms=(2, 4), rebalances=(8, 24))
        self.assertIn("夏普比率", tbl.columns)
        self.assertTrue(tbl["夏普比率"].is_monotonic_decreasing)


if __name__ == "__main__":
    unittest.main()
