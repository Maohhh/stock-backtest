"""
横截面多因子组合 —— 把年化做更高（且尽量稳健）
================================================

对比四套打法在同一篮子品种上的表现，并给出 **年化-杠杆前沿**：
  1. TS 动量（时序，单标的方向，作基准）
  2. XS 动量（横截面买强空弱，板块中性）
  3. Carry（展期结构买高空低，板块中性）
  4. 组合（XS动量 + Carry + TS动量 多因子等权）

核心观点：年化 ≈ 夏普 × 杠杆×波动。要更高的年化，先把 **夏普** 提上去（多因子、
横截面、板块中性、风险平价），再用 **目标波动率** 这个旋钮调杠杆。前沿表把
"想要多少年化、对应多大回撤" 一次列清楚。

用法：python scripts/run_factor_portfolio.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.panel import (
    carry_score,
    combine_scores,
    leverage_frontier,
    load_carry_panel,
    load_daily_panel,
    panel_backtest,
    score_to_weights,
    ts_momentum_score,
    xs_momentum_score,
)

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)
COST = 0.0005
GROSS = 2.0          # 总杠杆（sum|w|），再由目标波动率二次缩放
BASE_VOL = 0.15      # 主对比的目标波动


def _stat_line(tag, m):
    return (f"  {tag:16} 年化 {m['cagr']*100:6.1f}% | 夏普 {m['sharpe']:5.2f} | "
            f"回撤 {m['max_drawdown']*100:6.1f}% | Calmar {m['calmar']:4.2f} | "
            f"波动 {m['ann_vol']*100:4.1f}% | {m['years']:.1f}年")


def run():
    prices = load_daily_panel(start="2010-01-01")
    returns = prices.pct_change()
    # 只保留数据较全的品种（横截面需要足够宽度）
    good = returns.columns[returns.notna().sum() > 500]
    prices, returns = prices[good], returns[good]
    print(f"日线面板：{prices.shape[1]} 个品种，{prices.index.min().date()} ~ "
          f"{prices.index.max().date()}（{len(prices)} 日）")

    carry = load_carry_panel(start="2010-01-01")
    carry = carry.reindex(columns=[c for c in carry.columns if c in good])
    carry = carry.reindex(index=prices.index).ffill(limit=3)
    print(f"Carry 面板：{carry.notna().any().sum()} 个品种，"
          f"{carry.dropna(how='all').index.min().date()} ~ "
          f"{carry.dropna(how='all').index.max().date()}")

    # ---- 因子打分 ----
    ts = ts_momentum_score(prices)
    xs = xs_momentum_score(prices, sector_neutral=True)
    cy = carry_score(carry, sector_neutral=True)
    combo = combine_scores({"xs": xs, "carry": cy, "ts": ts})

    books = {
        "TS动量": ts,
        "XS动量(板块中性)": xs,
        "Carry(板块中性)": cy,
        "组合(XS+Carry+TS)": combo,
    }

    print("\n" + "=" * 78)
    print(f"四套打法对比（总杠杆 sum|w|={GROSS}，目标波动 {int(BASE_VOL*100)}%，"
          f"单边成本 {COST*1e4:.0f}bp）")
    print("=" * 78)
    results = {}
    for tag, score in books.items():
        w = score_to_weights(score, returns, gross=GROSS)
        m = panel_backtest(returns, w, cost=COST, target_vol=BASE_VOL)
        results[tag] = m
        print(_stat_line(tag, m))

    # ---- 年化-杠杆前沿（用最好的组合）----
    best = "组合(XS+Carry+TS)"
    w = score_to_weights(books[best], returns, gross=GROSS)
    m0 = panel_backtest(returns, w, cost=COST, target_vol=None)  # 不二次加杠杆，取原始收益
    front = leverage_frontier(m0["returns"])
    print("\n" + "=" * 78)
    print(f"年化-杠杆前沿（{best}，夏普≈{m0['sharpe']:.2f} 不变，调目标波动放大年化）")
    print("=" * 78)
    fd = front.copy()
    for c in ["目标波动", "年化", "最大回撤"]:
        fd[c] = fd[c].map(lambda v: f"{v*100:.0f}%" if c == "目标波动" else f"{v*100:.1f}%")
    fd["约杠杆"] = fd["约杠杆"].map(lambda v: f"{v:.1f}x")
    fd["夏普"] = fd["夏普"].map(lambda v: f"{v:.2f}")
    fd["Calmar"] = fd["Calmar"].map(lambda v: f"{v:.2f}")
    print(fd.to_string(index=False))

    _plot(results, m0, front)
    _write_md(results, m0, front, prices, carry)
    front.to_csv(RESULTS / "leverage_frontier.csv", index=False)
    print(f"\n✅ 已保存 {RESULTS}/factor_portfolio.png、leverage_frontier.csv、FACTOR_REPORT.md")
    return results, front


def _plot(results, m0, front):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["axes.unicode_minus"] = False
        labels = {"TS动量": "TS-mom", "XS动量(板块中性)": "XS-mom",
                  "Carry(板块中性)": "Carry", "组合(XS+Carry+TS)": "Combined"}
        fig, ax = plt.subplots(figsize=(11, 6))
        for tag, m in results.items():
            ax.plot(m["equity"].index, m["equity"].values,
                    label=f"{labels.get(tag, tag)} (Sh {m['sharpe']:.2f})",
                    lw=2 if "组合" in tag else 1.3)
        ax.set_yscale("log")
        ax.set_title("Cross-Sectional Multi-Factor Books (vol-targeted 15%)")
        ax.set_ylabel("Equity (log, start=1.0)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS / "factor_portfolio.png", dpi=110)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠️ 绘图跳过: {exc!r}")


def _write_md(results, m0, front, prices, carry):
    L = ["# 横截面多因子组合 —— 把年化做更高\n"]
    L.append(f"- 面板：{prices.shape[1]} 个主力连续品种，{prices.index.min().date()} ~ "
             f"{prices.index.max().date()}；单边成本 {COST*1e4:.0f}bp\n")
    L.append("## 核心结论\n")
    L.append("> 年化 ≈ **夏普 × 杠杆×波动**。单标的趋势夏普天花板约 0.6~0.7；换成 "
             "**横截面板块中性多空 + Carry + 风险平价** 后夏普明显抬升，再用目标波动率"
             "调杠杆，就能在可控回撤下把年化做高。\n")
    L.append("## 四套打法（目标波动 15%）\n")
    L.append("| 打法 | 年化 | 夏普 | 最大回撤 | Calmar |")
    L.append("|---|---|---|---|---|")
    for tag, m in results.items():
        L.append(f"| {tag} | {m['cagr']*100:.1f}% | {m['sharpe']:.2f} | "
                 f"{m['max_drawdown']*100:.1f}% | {m['calmar']:.2f} |")
    L.append("\n## 年化-杠杆前沿（组合，夏普不变，调目标波动）\n")
    L.append("| 目标波动 | 约杠杆 | 年化 | 夏普 | 最大回撤 | Calmar |")
    L.append("|---|---|---|---|---|---|")
    for _, r in front.iterrows():
        L.append(f"| {r['目标波动']*100:.0f}% | {r['约杠杆']:.1f}x | {r['年化']*100:.1f}% | "
                 f"{r['夏普']:.2f} | {r['最大回撤']*100:.1f}% | {r['Calmar']:.2f} |")
    L.append("\n## 诚实提示\n")
    L.append("- Carry 数据仅 2023-09 起（约 2.75 年），含 Carry 的组合在该段样本上夏普偏高，"
             "**短样本有高估风险**；XS 动量在 2010 年以来长样本上更可信。\n")
    L.append("- 高杠杆档（30%+ 波动）回撤会等比放大，且实盘有保证金、流动性、极端跳空风险，"
             "**不可照搬**。前沿表只说明 “想要更高年化要付出多大回撤” 的取舍。\n")
    L.append("- 参数（回看窗口、杠杆、中性化方式）未做严格滚动样本外，存在过拟合风险。\n")
    (RESULTS / "FACTOR_REPORT.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    run()
