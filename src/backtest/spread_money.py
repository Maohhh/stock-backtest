"""
跨品种套利篮子 —— 真实资金回测

用真实合约参数（乘数、保证金、最小变动）、真实手续费 + 滑点，按"对冲手数"成交，
把价差套利换算成**真金白银的 ¥ 盈亏曲线**，并给出占用保证金、年化收益率、最大回撤(¥)。

每个套利对以一组固定对冲手数为"1 个单位"成交；账户资金决定能持有多少个单位。
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..indicators.spread_zscore import spread_zscore, reversion_position
from ..data.contracts import list_contracts, _load


# 合约规格: 乘数(吨/手), 最小变动(元/吨), 保证金率, 往返手续费(元/手)
SPEC: Dict[str, dict] = {
    "C":  dict(mult=10,  tick=1.0, margin=0.09, comm=3.0),
    "CS": dict(mult=10,  tick=1.0, margin=0.09, comm=3.0),
    "RB": dict(mult=10,  tick=1.0, margin=0.10, comm=6.0),
    "HC": dict(mult=10,  tick=1.0, margin=0.10, comm=6.0),
    "I":  dict(mult=100, tick=0.5, margin=0.12, comm=15.0),
    "M":  dict(mult=10,  tick=1.0, margin=0.09, comm=3.0),
    "RM": dict(mult=10,  tick=1.0, margin=0.08, comm=3.0),
}


@dataclass
class SpreadSpec:
    name: str
    prod_a: str
    prod_b: str
    lots_a: int
    lots_b: int
    mode: str = "diff"        # 'diff' 或 'ratio'，仅用于生成 z 信号


def _round_trip_cost_yuan(spec: SpreadSpec) -> float:
    """一次完整套利交易（进+出，两条腿）的真实成本(元): 手续费 + 4 次成交各 1 跳滑点。"""
    a, b = SPEC[spec.prod_a], SPEC[spec.prod_b]
    comm = (a["comm"] * spec.lots_a + b["comm"] * spec.lots_b)
    slip = 2 * (a["tick"] * a["mult"] * spec.lots_a + b["tick"] * b["mult"] * spec.lots_b)
    return comm + slip


def _unit_margin_yuan(spec: SpreadSpec, pa: float, pb: float) -> float:
    """持有 1 个单位的占用保证金(元)。保守起见两条腿都计（实盘挂牌套利指令通常更省）。"""
    a, b = SPEC[spec.prod_a], SPEC[spec.prod_b]
    return (pa * a["mult"] * spec.lots_a * a["margin"]
            + pb * b["mult"] * spec.lots_b * b["margin"])


def backtest_spread_money(spec: SpreadSpec, window=30, entry=2.0, exit=0.5, stop=4.0,
                          data_dir="data/futures_contracts"):
    """
    对一个套利对做真实资金回测（1 个单位）。返回逐笔 ¥ 盈亏(按平仓日) 与统计。
    """
    a_specd, b_specd = SPEC[spec.prod_a], SPEC[spec.prod_b]
    ca = {(y, m): s for s, y, m, d in list_contracts(spec.prod_a, data_dir)}
    cb = {(y, m): s for s, y, m, d in list_contracts(spec.prod_b, data_dir)}
    dmap = {(y, m): d for s, y, m, d in list_contracts(spec.prod_a, data_dir)}

    daily: Dict[pd.Timestamp, float] = {}
    margins: List[float] = []
    pnls: List[float] = []
    cost = _round_trip_cost_yuan(spec)

    for key in sorted(set(ca) & set(cb)):
        sa, sb = _load(ca[key], data_dir), _load(cb[key], data_dir)
        if sa is None or sb is None:
            continue
        df = pd.DataFrame({"a": sa, "b": sb}).dropna()
        df = df[df.index < dmap[key] - pd.Timedelta(days=20)]
        if len(df) < window + 10:
            continue
        if spec.mode == "ratio":
            sp = (np.log(df["b"]) - np.log(df["a"])) * 10000
        else:
            sp = df["b"] - df["a"]
        z = spread_zscore(sp, window)
        pos = reversion_position(z, entry, exit, stop)
        prev, ea, eb = 0, None, None
        for i in range(len(pos)):
            cur = pos.iloc[i]
            if prev == 0 and cur != 0:
                ea, eb = df["a"].iloc[i], df["b"].iloc[i]
                margins.append(_unit_margin_yuan(spec, ea, eb))
            elif prev != 0 and cur != prev:
                xa, xb = df["a"].iloc[i], df["b"].iloc[i]
                # 多价差(+1)=多 b 空 a
                pnl = prev * (spec.lots_b * b_specd["mult"] * (xb - eb)
                              - spec.lots_a * a_specd["mult"] * (xa - ea)) - cost
                daily[df.index[i]] = daily.get(df.index[i], 0.0) + pnl
                pnls.append(pnl)
                if cur != 0:
                    ea, eb = xa, xb
                    margins.append(_unit_margin_yuan(spec, ea, eb))
            prev = cur

    series = pd.Series(daily).sort_index()
    p = np.array(pnls)
    stats = dict(
        n_trades=len(p),
        win_rate=float((p > 0).mean()) if len(p) else 0.0,
        total_yuan=float(p.sum()),
        avg_yuan=float(p.mean()) if len(p) else 0.0,
        unit_margin=float(np.median(margins)) if margins else 0.0,
        cost_per_trade=cost,
    )
    return series, stats
