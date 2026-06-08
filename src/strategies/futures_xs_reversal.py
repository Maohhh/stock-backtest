"""
期货横截面反转策略

把 indicators.xs_reversal 的目标权重，封装成一个面向板块面板的策略对象，
并通过 backtest.futures_engine 完成回测。提供参数搜索与波动率目标接口。
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..indicators.xs_reversal import target_weights
from ..backtest.futures_engine import run_panel_backtest, FuturesBacktestResult, bars_per_year


@dataclass
class FuturesXSReversalStrategy:
    """
    板块内横截面反转策略。

    参数:
        form: 形成期（K 线根数），衡量短期相对强弱
        rebalance: 再平衡间隔（K 线根数），控制换手
        cost: 单边交易成本（小数）
        target_vol: 波动率目标（年化，小数）；None 则不加杠杆
    """
    form: int = 4
    rebalance: int = 24
    cost: float = 0.0002
    target_vol: Optional[float] = None

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        return target_weights(prices, form=self.form, rebalance=self.rebalance)

    def run(self, prices: pd.DataFrame) -> FuturesBacktestResult:
        return run_panel_backtest(prices, self.weights(prices),
                                  cost=self.cost, target_vol=self.target_vol)


def optimize(prices: pd.DataFrame, forms=(1, 2, 4), rebalances=(4, 8, 16, 24),
             cost: float = 0.0002) -> pd.DataFrame:
    """
    在给定面板上网格搜索 (form, rebalance)，按夏普排序返回结果表。

    仅作研究参考——短样本上务必结合样本外检验，避免过拟合。
    """
    rows = []
    for f in forms:
        for q in rebalances:
            res = FuturesXSReversalStrategy(form=f, rebalance=q, cost=cost).run(prices)
            rows.append({"form": f, "rebalance": q, **res.summary()})
    return pd.DataFrame(rows).sort_values("夏普比率", ascending=False).reset_index(drop=True)
