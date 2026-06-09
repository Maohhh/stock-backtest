"""
跨品种套利篮子 (Cross-Product Spread Arbitrage Basket)

把多个**低相关**的同月跨品种价差（玉米-淀粉、螺纹-热卷、铁矿-螺纹、豆粕-菜粕……）
各自做 Z 分数回归，再按风险平价组合。低相关性带来强分散：组合夏普显著高于任一单腿。

这是对"只做套利会不会更好"的回答——**做一篮子跨品种套利**最稳：高胜率、低回撤、可分散。
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from ..indicators.spread_zscore import spread_zscore, reversion_position


def backtest_segments(segments: List[pd.Series], window: int = 30, entry: float = 2.0,
                      exit: float = 0.5, stop: float = 4.0, cost: float = 4.0):
    """
    对一组价差段做 Z 回归回测，返回 (逐笔 PnL Series[按平仓日], 交易统计 dict)。
    """
    daily: Dict[pd.Timestamp, float] = {}
    pnls = []
    for sp in segments:
        if len(sp) < window + 5:
            continue
        z = spread_zscore(sp, window)
        pos = reversion_position(z, entry, exit, stop)
        prev, ep = 0, None
        for i in range(len(pos)):
            cur = pos.iloc[i]
            if prev == 0 and cur != 0:
                ep = sp.iloc[i]
            elif prev != 0 and cur != prev:
                pnl = (sp.iloc[i] - ep) * prev - cost
                dt = sp.index[i]
                daily[dt] = daily.get(dt, 0.0) + pnl
                pnls.append(pnl)
                if cur != 0:
                    ep = sp.iloc[i]
            prev = cur
    pnl_arr = np.array(pnls, dtype=float)
    stats = _trade_stats(pnl_arr)
    series = pd.Series(daily).sort_index()
    return series, stats


def _trade_stats(p: np.ndarray) -> dict:
    if len(p) == 0:
        return dict(n_trades=0, win_rate=0.0, payoff=float("nan"),
                    profit_factor=float("nan"), total=0.0)
    w, l = p[p > 0], p[p < 0]
    return dict(
        n_trades=len(p),
        win_rate=float((p > 0).mean()),
        payoff=float(w.mean() / -l.mean()) if len(w) and len(l) else float("nan"),
        profit_factor=float(w.sum() / -l.sum()) if len(l) else float("inf"),
        total=float(p.sum()),
    )


@dataclass
class SpreadBasket:
    """
    跨品种套利篮子。

    参数:
        legs: {名称: 价差段列表}（用 data.contracts.matched_spread_segments 生成）
        window/entry/exit/stop/cost: z 回归参数
    """
    legs: Dict[str, List[pd.Series]]
    window: int = 30
    entry: float = 2.0
    exit: float = 0.5
    stop: float = 4.0
    cost: float = 4.0

    def run(self) -> dict:
        leg_pnl, leg_stats = {}, {}
        for name, segs in self.legs.items():
            series, st = backtest_segments(segs, self.window, self.entry,
                                           self.exit, self.stop, self.cost)
            if st["n_trades"] >= 5:
                leg_pnl[name] = series
                leg_stats[name] = st
        # 月度对齐 + 风险平价（按月度波动归一后等权）
        monthly = pd.DataFrame({k: v.groupby(pd.Grouper(freq="ME")).sum()
                                for k, v in leg_pnl.items()}).fillna(0.0)
        risk_norm = monthly / monthly.std()
        combo = risk_norm.mean(axis=1)
        ann_sharpe = float(combo.mean() / combo.std() * np.sqrt(12)) if combo.std() > 0 else 0.0
        eq = combo.cumsum()
        return {
            "leg_stats": leg_stats,
            "corr": monthly.corr(),
            "monthly": monthly,
            "combo_monthly": combo,
            "combo_equity": eq,
            "combo_ann_sharpe": ann_sharpe,
            "combo_win_month": float((combo > 0).mean()),
            "leg_ann_sharpe": {c: float(monthly[c].mean() / monthly[c].std() * np.sqrt(12))
                               for c in monthly.columns if monthly[c].std() > 0},
        }
