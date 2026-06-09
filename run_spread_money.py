"""
跨品种套利篮子 —— 真实资金回测（看到底能赚多少钱）

按真实合约参数 + 你的手续费 + 滑点，用对冲手数成交，给出 ¥ 盈亏曲线、
占用保证金、年化收益率、最大回撤(¥)。

用法: python run_spread_money.py [账户资金, 默认200000]
"""

import os
import sys

import numpy as np
import pandas as pd

from src.data.contracts import download_product_contracts
from src.backtest.spread_money import SpreadSpec, backtest_spread_money

# 套利对（全部为券商挂牌单一套利标的, 交易所比例 1:1, diff 价差）
# 精选"收益/回撤比"最优的 3 个核心对，分属淀粉/塑料/棉纺三个不相关板块
SPREADS = [
    SpreadSpec("玉米-淀粉", "C", "CS", lots_a=1, lots_b=1, mode="diff"),
    SpreadSpec("PVC-聚丙烯", "V", "PP", lots_a=1, lots_b=1, mode="diff"),
    SpreadSpec("棉花-棉纱", "CF", "CY", lots_a=1, lots_b=1, mode="diff"),
]
UTIL = 0.5   # 每个套利对最多用掉其资金额度的 50%（留安全垫）


def ensure_data():
    if not os.path.isdir("data/futures_contracts") or len(os.listdir("data/futures_contracts")) < 80:
        for p in ["C", "CS", "V", "PP", "CF", "CY"]:
            download_product_contracts(p)


def main():
    capital = float(sys.argv[1]) if len(sys.argv) > 1 else 200000.0
    ensure_data()

    legs, stats = {}, {}
    for s in SPREADS:
        ser, st = backtest_spread_money(s)
        legs[s.name], stats[s.name] = ser, st

    print("=" * 70)
    print(f"跨品种套利篮子 · 真实资金回测  账户资金 ¥{capital:,.0f}")
    print("=" * 70)
    print(f"{'套利对':<12}{'手数(A:B)':>9}{'笔数':>5}{'胜率':>6}{'单位保证金':>10}{'单位总盈利':>11}{'成本/次':>8}")
    alloc = capital / len(SPREADS)
    units = {}
    for s in SPREADS:
        st = stats[s.name]
        u = max(1, int(alloc * UTIL / st["unit_margin"])) if st["unit_margin"] > 0 else 1
        units[s.name] = u
        print(f"{s.name:<12}{f'{s.lots_a}:{s.lots_b}':>9}{st['n_trades']:>5}{st['win_rate']*100:>5.0f}%"
              f"{st['unit_margin']:>10,.0f}{st['total_yuan']:>11,.0f}{st['cost_per_trade']:>8,.0f}")

    daily = pd.DataFrame({n: legs[n] for n in legs}).fillna(0.0).sort_index()
    years = (daily.index[-1] - daily.index[0]).days / 365.25
    n_total = sum(stats[n]['n_trades'] for n in stats)

    def equity_for(util):
        u = {s.name: max(1, int(alloc * util / stats[s.name]["unit_margin"]))
             for s in SPREADS if stats[s.name]["unit_margin"] > 0}
        scaled = sum(daily[n] * u[n] for n in daily.columns).sort_index()
        eq = capital + scaled.cumsum()
        peak = eq.cummax()
        return u, eq, scaled.sum(), ((eq - peak) / peak).min()

    print("\n[资金利用率 / 收益 / 回撤 前沿]  (越激进越赚但回撤越大)")
    print(f"{'额度利用率':>10}{'总盈利':>12}{'总收益率':>9}{'年化(简单)':>10}{'最大回撤':>9}")
    for util in [0.5, 1.0, 1.5, 2.0]:
        u, eq, tot, dd = equity_for(util)
        print(f"{util*100:>9.0f}%{tot:>12,.0f}{tot/capital*100:>8.0f}%"
              f"{tot/capital/years*100:>9.1f}%{-dd*100:>8.1f}%")

    util = 1.0   # 推荐：满额度（约 1x，仍有空仓垫）
    units, equity, total_profit, max_dd_pct = equity_for(util)
    print("\n推荐档(满额度~1x)持仓单位数:", {n: units[n] for n in units})
    print("-" * 70)
    print(f"  回测区间: {equity.index[0].date()} ~ {equity.index[-1].date()}  ({years:.1f} 年)")
    print(f"  总盈利:   ¥{total_profit:,.0f}   (本金 ¥{capital:,.0f} -> ¥{equity.iloc[-1]:,.0f})")
    print(f"  简单年化: {total_profit/capital/years*100:.1f}%/年   最大回撤: {-max_dd_pct*100:.1f}%")
    print(f"  年均交易: {n_total/years:.0f} 次/年（{len(SPREADS)} 个套利对合计）")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(equity.index, equity.values, color="darkgreen", lw=2,
                label=f"Account equity: +¥{total_profit:,.0f} ({total_profit/capital*100:.0f}%), "
                      f"maxDD {-max_dd_pct*100:.0f}%")
        ax.axhline(capital, color="gray", ls="--", lw=0.8)
        ax.set_title(f"Spread arbitrage basket - real money equity (start ¥{capital:,.0f})")
        ax.set_ylabel("Account equity (CNY)"); ax.legend(); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig("data/spread_money_equity.png", dpi=110)
        print("\n净值图已保存: data/spread_money_equity.png")
    except Exception as e:
        print(f"（绘图跳过: {e}）")


if __name__ == "__main__":
    main()
