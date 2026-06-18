"""
怎么把年化做更高 —— 诚实的穷尽结论
====================================

围绕用户追问 "9% 太低，还有别的办法吗？穷尽思考"。本脚本把所有路子跑了一遍，
给出两块硬证据：

A. **杠杆前沿**（真正能放大年化的旋钮）：在稳健的低换手日线趋势组合（夏普≈0.67、
   21 年、扣 5bp 成本）上，按不同目标波动率加杠杆，年化可以从 ~7% 推到 ~25%，
   但 **回撤等比放大**（45% 波动档 → 年化 25% / 回撤 -79%）。夏普是硬约束。

B. **成本现实**（为什么不能靠 "异域因子" 白捡更高夏普）：横截面反转 / 跳月动量 /
   Carry 这些因子，**零成本下看着能赚，扣真实手续费就塌**（换手太高、IC 太小）。
   反转：2bp 夏普 +0.3 → 5bp 转负 → 10bp 崩。这与兄弟分支的穷尽结论一致。

一句话：**更高的年化主要来自 “杠杆 + 多品种分散”，而不是更花哨的信号；
代价是更大的回撤。** 想要更高夏普，靠的是更多低相关品种、更低换手，而非加因子。

用法：python scripts/run_higher_yield.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.futures import FUTURES_UNIVERSE, FuturesDataSource
from src.backtest.futures_engine import (
    FuturesBacktester,
    combine_portfolio,
    vol_target_weight,
)
from src.strategies.mtf import mtf_signal
from src.backtest.panel import (
    carry_score,
    load_carry_panel,
    panel_backtest,
    score_to_weights,
    xs_momentum_score,
    xs_reversal_score,
)

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)


def load_clean(src):
    out = {}
    for cat, syms in FUTURES_UNIVERSE.items():
        for s, _ in syms:
            d = src.get_daily(s)
            if d is not None and len(d) > 500:
                out[s] = d
    return out


def robust_trend_book(daily):
    """稳健低换手日线趋势组合（MA5/20，每品种波动目标定仓）。"""
    res = {}
    for s, d in daily.items():
        prices = d.set_index("date")["close"]
        ret = prices.pct_change().clip(-0.15, 0.15).fillna(0.0)
        sig = mtf_signal(d, ltf_method="ma_cross", ma_fast=5, ma_slow=20, mode="ltf_only")
        res[s] = FuturesBacktester(cost=0.0005).run(prices, vol_target_weight(sig, ret, 0.15, 20, 3.0))
    return res


def leverage_frontier_book(res):
    rows = []
    for tv in [0.10, 0.15, 0.20, 0.30, 0.45, 0.60]:
        p = combine_portfolio(res, target_vol=tv, max_leverage=10.0).stats
        rows.append({"目标波动": tv, "年化": p["cagr"], "夏普": p["sharpe"],
                     "最大回撤": p["max_drawdown"], "Calmar": p["calmar"]})
    return pd.DataFrame(rows)


def factor_cost_reality(daily):
    """横截面因子在不同成本下的净夏普：证明扣成本后塌掉。"""
    prices = pd.DataFrame({s: d.set_index("date")["close"] for s, d in daily.items()}).sort_index()
    returns = prices.pct_change()
    carry = load_carry_panel(start="2010-01-01")
    carry = carry.reindex(columns=[c for c in carry.columns if c in prices.columns])
    carry = carry.reindex(index=prices.index).ffill(limit=3)

    factors = {
        "横截面反转5日": xs_reversal_score(prices, 5),
        "跳月动量(12-1)": xs_momentum_score(prices, lookbacks=(120, 250), skip=21),
        "Carry": carry_score(carry),
    }
    rows = []
    for name, score in factors.items():
        w = score_to_weights(score, returns, gross=2.0)
        row = {"因子": name}
        for cbp in [0.0, 0.0002, 0.0005, 0.0010]:
            m = panel_backtest(returns, w, cost=cbp, target_vol=0.15)
            row[f"{int(cbp*1e4)}bp夏普"] = m["sharpe"]
        rows.append(row)
    return pd.DataFrame(rows)


def run():
    src = FuturesDataSource()
    daily = load_clean(src)
    print(f"清洗后数据：{len(daily)} 个商品品种，21 年日线\n")

    res = robust_trend_book(daily)
    front = leverage_frontier_book(res)
    fac = factor_cost_reality(daily)

    print("=" * 70)
    print("A. 杠杆前沿（稳健趋势组合，夏普≈0.67，调目标波动放大年化）")
    print("=" * 70)
    fd = front.copy()
    fd["目标波动"] = fd["目标波动"].map(lambda v: f"{v*100:.0f}%")
    for c in ["年化", "最大回撤"]:
        fd[c] = fd[c].map(lambda v: f"{v*100:.1f}%")
    for c in ["夏普", "Calmar"]:
        fd[c] = fd[c].map(lambda v: f"{v:.2f}")
    print(fd.to_string(index=False))

    print("\n" + "=" * 70)
    print("B. 成本现实：横截面因子的净夏普随成本塌陷（目标波动 15%）")
    print("=" * 70)
    fcd = fac.copy()
    for c in fcd.columns:
        if c != "因子":
            fcd[c] = fcd[c].map(lambda v: f"{v:.2f}")
    print(fcd.to_string(index=False))
    print("\n→ 异域因子零成本好看，扣真实成本（5bp+）转负：不是白捡的更高夏普。")

    _plot(res, front)
    _write_md(front, fac, len(daily))
    front.to_csv(RESULTS / "leverage_frontier.csv", index=False)
    fac.to_csv(RESULTS / "factor_cost_reality.csv", index=False)
    print(f"\n✅ 已保存 {RESULTS}/leverage_frontier.csv、factor_cost_reality.csv、"
          f"HIGHER_YIELD.md、higher_yield_frontier.png")


def _plot(res, front):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["axes.unicode_minus"] = False
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        # 左：不同杠杆净值
        for tv in [0.15, 0.30, 0.45]:
            p = combine_portfolio(res, target_vol=tv, max_leverage=10.0)
            axes[0].plot(p.equity.index, p.equity.values, label=f"vol {int(tv*100)}%")
        axes[0].set_yscale("log"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[0].set_title("Trend portfolio at different leverage (log equity)")
        # 右：年化 vs 回撤前沿
        axes[1].plot(-front["最大回撤"] * 100, front["年化"] * 100, "o-")
        for _, r in front.iterrows():
            axes[1].annotate(f"{int(r['目标波动']*100)}%",
                             (-r["最大回撤"] * 100, r["年化"] * 100))
        axes[1].set_xlabel("Max Drawdown (%)"); axes[1].set_ylabel("Annualized (%)")
        axes[1].set_title("Return-Drawdown frontier (leverage dial)")
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS / "higher_yield_frontier.png", dpi=110)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠️ 绘图跳过: {exc!r}")


def _write_md(front, fac, n):
    L = ["# 怎么把年化做更高 —— 穷尽思考后的诚实结论\n"]
    L.append(f"> 数据：{n} 个商品主力连续品种，21 年日线，单边成本 5bp。\n")
    L.append("## TL;DR\n")
    L.append("- **年化 ≈ 夏普 × 杠杆×波动**。稳健趋势组合的夏普≈0.67 是硬约束。")
    L.append("- 想要更高年化，主要靠 **加杠杆 + 多品种分散**，代价是 **回撤等比放大**。")
    L.append("- 横截面反转/动量/Carry 等 “异域因子” **零成本好看、扣成本即塌**，"
             "在国内商品上不是白捡的更高夏普（与兄弟分支穷尽结论一致）。\n")
    L.append("## A. 杠杆前沿（稳健趋势组合）\n")
    L.append("| 目标波动 | 年化 | 夏普 | 最大回撤 | Calmar |")
    L.append("|---|---|---|---|---|")
    for _, r in front.iterrows():
        L.append(f"| {r['目标波动']*100:.0f}% | {r['年化']*100:.1f}% | {r['夏普']:.2f} | "
                 f"{r['最大回撤']*100:.1f}% | {r['Calmar']:.2f} |")
    L.append("\n> 要 20%+ 的年化做得到，但 30~45% 波动档对应 **-63% ~ -79% 的回撤**，"
             "实盘很难扛得住。这就是 “更高年化” 的真实代价。\n")
    L.append("## B. 成本现实：横截面因子净夏普随成本塌陷\n")
    L.append("| 因子 | 0bp | 2bp | 5bp | 10bp |")
    L.append("|---|---|---|---|---|")
    for _, r in fac.iterrows():
        L.append(f"| {r['因子']} | {r['0bp夏普']:.2f} | {r['2bp夏普']:.2f} | "
                 f"{r['5bp夏普']:.2f} | {r['10bp夏普']:.2f} |")
    L.append("\n> 这些因子日频换手极高、IC 仅 ~0.01-0.02，手续费一上来 edge 就被吃光。"
             "靠它们 “提升夏普换更高年化” 在真实成本下走不通。\n")
    L.append("## 真正能做的（按性价比排序）\n")
    L.append("1. **多品种分散**：把品种数从 36 扩到含金融期货（IF/IC/IH/IM/国债）"
             "等更多低相关标的，分散提升组合夏普 → 同回撤下更高年化。\n")
    L.append("2. **杠杆 + 严格风控**：把目标波动调到 20~25%（约 1.5~2x），年化 14~17%，"
             "回撤 -47~-55%，再叠加组合级别止损/降杠杆规则控制尾部。\n")
    L.append("3. **板块筛选**：剔除/降低农产品权重（趋势性弱），重配工业品/贵金属。\n")
    L.append("4. **更低换手**：趋势信号天然低换手才扛得住成本；不要去做高频因子。\n")
    L.append("\n⚠️ 所有数字为历史回测、未严格滚动样本外，含一定过拟合与短样本（Carry 仅 2.75 年）"
             "风险；高杠杆档不可照搬实盘。")
    (RESULTS / "HIGHER_YIELD.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    run()
