"""
单标的趋势集成指标 / 趋势跟踪策略 单元测试
"""

import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indicators.trend_ensemble import efficiency_ratio, trend_signal
from src.strategies.trend_following import TrendFollowingStrategy


def _make(trend=True, n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    if trend:
        close = 1000 + np.cumsum(rng.normal(0.8, 1.0, n))   # 强上行趋势
    else:
        close = 1000 + np.cumsum(rng.normal(0.0, 1.0, n))   # 无漂移震荡
        close = 1000 + (close - close.mean()) * 0.3          # 压扁成区间
    df = pd.DataFrame({"open": close, "high": close * 1.005,
                       "low": close * 0.995, "close": close}, index=idx)
    return df


class TestTrend(unittest.TestCase):

    def test_efficiency_ratio_range(self):
        df = _make(trend=True)
        er = efficiency_ratio(df["close"], 20).dropna()
        self.assertTrue((er >= 0).all() and (er <= 1.0001).all())
        # 强趋势的 ER 应明显高于震荡
        er_choppy = efficiency_ratio(_make(trend=False)["close"], 20).dropna()
        self.assertGreater(er.mean(), er_choppy.mean())

    def test_signal_in_range(self):
        sig = trend_signal(_make(trend=True))
        self.assertTrue((sig.dropna().abs() <= 1.0).all())

    def test_signal_long_in_uptrend(self):
        sig = trend_signal(_make(trend=True)).dropna()
        # 上行趋势里多头信号应占多数
        self.assertGreater((sig > 0).mean(), (sig < 0).mean())

    def test_choppy_flat_more(self):
        # 震荡市里 ER 过滤应让空仓(0)占比更高
        s_tr = trend_signal(_make(trend=True))
        s_ch = trend_signal(_make(trend=False))
        self.assertGreaterEqual((s_ch == 0).mean(), (s_tr == 0).mean())

    def test_strategy_runs(self):
        res = TrendFollowingStrategy(target_vol=0.3, cost=0.0002).run(_make(trend=True), "T")
        self.assertTrue(np.isfinite(res.sharpe))
        self.assertLessEqual(res.max_drawdown, 0.0)
        self.assertEqual(res.bars_per_year, 252)

    def test_leverage_cap(self):
        strat = TrendFollowingStrategy(target_vol=0.3, max_leverage=2.0)
        pos = strat.position(_make(trend=True)).dropna()
        self.assertLessEqual(pos.abs().max(), 2.0 + 1e-9)

    def test_higher_vol_target_higher_return_magnitude(self):
        df = _make(trend=True)
        # max_leverage 设高，避免杠杆上限把两档都顶到同一水平
        lo = TrendFollowingStrategy(target_vol=0.15, cost=0.0, max_leverage=50).run(df, "T").ann_vol
        hi = TrendFollowingStrategy(target_vol=0.40, cost=0.0, max_leverage=50).run(df, "T").ann_vol
        self.assertGreater(hi, lo)


if __name__ == "__main__":
    unittest.main()
