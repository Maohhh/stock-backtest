"""
真实资金回测 成本/保证金 计算 单元测试（纯函数，无需下载数据）
"""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.spread_money import (SpreadSpec, SPEC,
                                       _round_trip_cost_yuan, _unit_margin_yuan)


class TestSpreadMoney(unittest.TestCase):

    def test_specs_present(self):
        for p in ["C", "CS", "RB", "HC", "I", "M", "RM"]:
            self.assertIn(p, SPEC)
            self.assertGreater(SPEC[p]["mult"], 0)

    def test_round_trip_cost_corn_starch(self):
        # 玉米-淀粉 1:1: 手续费 3+3=6, 滑点 2*(1*10*1 + 1*10*1)=40 -> 46 元
        s = SpreadSpec("玉米-淀粉", "C", "CS", 1, 1)
        self.assertAlmostEqual(_round_trip_cost_yuan(s), 46.0, places=1)

    def test_round_trip_cost_iron_rebar(self):
        # I(100t,tick0.5,comm15) 1手 + RB(10t,tick1,comm6) 2手
        # 手续费 15 + 6*2=27; 滑点 2*(0.5*100*1 + 1*10*2)=2*(50+20)=140 -> 167
        s = SpreadSpec("铁矿-螺纹", "I", "RB", 1, 2, mode="ratio")
        self.assertAlmostEqual(_round_trip_cost_yuan(s), 167.0, places=1)

    def test_unit_margin_positive(self):
        s = SpreadSpec("玉米-淀粉", "C", "CS", 1, 1)
        m = _unit_margin_yuan(s, 2330, 2720)
        # 2330*10*0.09 + 2720*10*0.09 = 2097 + 2448 = 4545
        self.assertAlmostEqual(m, 2330 * 10 * 0.09 + 2720 * 10 * 0.09, places=0)
        self.assertGreater(m, 0)

    def test_more_lots_more_cost(self):
        s1 = SpreadSpec("x", "C", "CS", 1, 1)
        s2 = SpreadSpec("x", "C", "CS", 2, 2)
        self.assertGreater(_round_trip_cost_yuan(s2), _round_trip_cost_yuan(s1))


if __name__ == "__main__":
    unittest.main()
