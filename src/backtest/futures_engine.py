"""
期货收益率型回测引擎
====================

与仓库里偏股票（只做多、按股数）的 ``BacktestEngine`` 不同，期货回测需要：

1. **多空双向** —— 信号 ∈ [-1, +1]，负数表示做空。
2. **杠杆 / 保证金** —— 商品期货天然带杠杆，用 *目标波动率* 来决定仓位
   大小，使不同品种的年化收益可比、杠杆显性化。
3. **连续合约收益** —— 用收盘价百分比收益计算损益，规避主力换月跳空。

这是一个**收益率型（returns-based）**回测器：给定每根 K 线的目标权重
``weight``（决定于当根收盘、作用于下一根收益），输出净值曲线与绩效。

绩效指标里最关键的是 **年化收益率 (CAGR)**、**夏普**、**最大回撤** 和
**Calmar**，正好对应用户关心的 "更高的年化率"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class FuturesResult:
    equity: pd.Series          # 净值曲线（起点 1.0）
    returns: pd.Series         # 每根 K 线策略收益
    weight: pd.Series          # 实际持仓权重（已含杠杆，正多负空）
    stats: Dict[str, float] = field(default_factory=dict)

    def __getitem__(self, k):   # 方便 result['cagr'] 取指标
        return self.stats[k]


def annualize_factor(freq: str) -> float:
    return {"D": 252.0, "W": 52.0, "M": 12.0, "H1": 252 * 4.0}.get(freq, 252.0)


def vol_target_weight(
    signal: pd.Series,
    returns: pd.Series,
    target_vol: float = 0.15,
    vol_window: int = 20,
    max_leverage: float = 3.0,
    ann: float = 252.0,
) -> pd.Series:
    """把方向信号 (-1/0/+1 或连续值) 转成波动率目标权重。

    weight = signal * target_vol / realized_vol，并用 max_leverage 截断。
    realized_vol 用过去 ``vol_window`` 根收益的年化标准差（前移一位避免未来函数）。
    """
    realized = returns.rolling(vol_window).std().shift(1) * np.sqrt(ann)
    realized = realized.replace(0, np.nan)
    scale = (target_vol / realized).clip(upper=max_leverage)
    weight = (signal * scale).clip(-max_leverage, max_leverage)
    return weight.fillna(0.0)


class FuturesBacktester:
    """收益率型期货回测器。

    参数
    ----
    cost : float
        单边交易成本（手续费 + 滑点），以权重换手计费，默认万分之三 + 万分之二。
    freq : str
        K 线频率，用于年化（'D' 日线 / 'W' 周线 / 'H1' 等）。
    """

    def __init__(self, cost: float = 0.0005, freq: str = "D",
                 ret_clip: float = 0.15, ann: float | None = None):
        self.cost = cost
        self.freq = freq
        self.ann = ann if ann is not None else annualize_factor(freq)
        # 主力连续合约换月处会有跳空，单日 |收益| 超过该阈值视为拼接假信号并截断，
        # 避免把换月跳空（而非真实盈亏）计入策略损益。
        self.ret_clip = ret_clip

    def run(self, prices: pd.Series, weight: pd.Series) -> FuturesResult:
        """
        prices : 收盘价序列（index 为日期）
        weight : 目标权重，决定于该根收盘，作用于下一根收益（内部自动 shift）
        """
        prices = prices.astype(float)
        ret = prices.pct_change().clip(-self.ret_clip, self.ret_clip).fillna(0.0)
        weight = weight.reindex(prices.index).fillna(0.0)

        # t 根收盘定的仓位，吃 t+1 收益 -> 用 shift(1)
        eff_w = weight.shift(1).fillna(0.0)
        gross = eff_w * ret

        # 换手成本：当根权重相对上一根的变化
        turnover = weight.diff().abs().fillna(weight.abs())
        cost = turnover * self.cost

        strat_ret = gross - cost
        equity = (1.0 + strat_ret).cumprod()

        stats = self._metrics(strat_ret, equity, weight, turnover)
        return FuturesResult(equity=equity, returns=strat_ret, weight=weight, stats=stats)

    def _metrics(self, ret: pd.Series, equity: pd.Series,
                 weight: pd.Series, turnover: pd.Series) -> Dict[str, float]:
        n = len(ret)
        if n == 0 or equity.iloc[-1] <= 0:
            return {"cagr": -1.0, "sharpe": 0.0, "max_drawdown": -1.0,
                    "calmar": 0.0, "ann_vol": 0.0, "exposure": 0.0,
                    "turnover": 0.0, "win_rate": 0.0, "years": 0.0}
        years = n / self.ann
        cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else 0.0
        ann_vol = ret.std() * np.sqrt(self.ann)
        sharpe = ret.mean() / ret.std() * np.sqrt(self.ann) if ret.std() > 0 else 0.0
        downside = ret[ret < 0].std()
        sortino = ret.mean() / downside * np.sqrt(self.ann) if downside and downside > 0 else 0.0

        dd = equity / equity.cummax() - 1.0
        max_dd = dd.min()
        calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

        active = ret[weight.shift(1).fillna(0) != 0]
        win_rate = (active > 0).mean() if len(active) else 0.0

        return {
            "cagr": float(cagr),
            "ann_vol": float(ann_vol),
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "max_drawdown": float(max_dd),
            "calmar": float(calmar),
            "exposure": float(weight.abs().mean()),
            "turnover": float(turnover.sum() / years),  # 年化换手
            "win_rate": float(win_rate),
            "years": float(years),
            "final_equity": float(equity.iloc[-1]),
        }


def combine_portfolio(results: Dict[str, FuturesResult],
                      freq: str = "D",
                      target_vol: float | None = None,
                      max_leverage: float = 3.0) -> FuturesResult:
    """把多个品种的策略收益等权合成一个组合（每日 rebalance）。

    分散化让组合夏普显著高于单品种。等权平均会把波动除以 ~sqrt(N)，因此
    若给定 ``target_vol``，再按组合层面的滚动实现波动率把整体杠杆放大到目标
    波动，使年化收益反映一个真实可交易的杠杆 CTA（夏普不变、收益等比放大）。
    """
    if not results:
        raise ValueError("no results to combine")
    ann = annualize_factor(freq)
    ret_df = pd.DataFrame({k: v.returns for k, v in results.items()}).sort_index()
    # 等权：对当根有数据的品种取均值（NaN 不计入）
    port_ret = ret_df.mean(axis=1, skipna=True).fillna(0.0)

    if target_vol is not None:
        realized = port_ret.rolling(40).std().shift(1) * np.sqrt(ann)
        lev = (target_vol / realized.replace(0, np.nan)).clip(upper=max_leverage).fillna(0.0)
        port_ret = (lev * port_ret).fillna(0.0)

    equity = (1.0 + port_ret).cumprod()
    weight_df = pd.DataFrame({k: v.weight for k, v in results.items()})
    avg_w = weight_df.abs().mean(axis=1).fillna(0.0)

    bt = FuturesBacktester(freq=freq)
    turnover = pd.Series(0.0, index=port_ret.index)
    stats = bt._metrics(port_ret, equity, avg_w, turnover)
    return FuturesResult(equity=equity, returns=port_ret, weight=avg_w, stats=stats)
