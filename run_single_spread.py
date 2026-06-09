"""
单一套利标的 · 本金杠杆前沿（专做玉米-淀粉这类干净价差）

只交易**一个**跨品种套利标的（默认玉米-淀粉 C/CS），用真实合约参数+你的成本，
展示给定本金下"持仓手数 / 杠杆 / 年化 / 最大回撤"的前沿，以及净值曲线。

玉米-淀粉价差历史"收益/回撤比"极高(≈10)，适合作为单标的重仓对象；
但单标的无分散，务必看清回撤、用 4σ 止损与账户熔断。

用法: python run_single_spread.py [品种A] [品种B] [本金]
  例: python run_single_spread.py C CS 200000
"""

import os
import sys

import numpy as np
import pandas as pd

from src.data.contracts import download_product_contracts
from src.backtest.spread_money import SpreadSpec, backtest_spread_money, SPEC


def main():
    a = sys.argv[1] if len(sys.argv) > 1 else "C"
    b = sys.argv[2] if len(sys.argv) > 2 else "CS"
    cap = float(sys.argv[3]) if len(sys.argv) > 3 else 200000.0
    name = f"{a}-{b}"

    if not os.path.isdir("data/futures_contracts") or \
       len([f for f in os.listdir("data/futures_contracts") if f.startswith(a)]) < 3:
        for p in (a, b):
            download_product_contracts(p)

    ser, st = backtest_spread_money(SpreadSpec(name, a, b, 1, 1, "diff"))
    yrs = (ser.index[-1] - ser.index[0]).days / 365.25
    um = st["unit_margin"]

    print("=" * 66)
    print(f"单一套利标的 · {name}   本金 ¥{cap:,.0f}")
    print(f"区间 {ser.index[0].date()} ~ {ser.index[-1].date()}  ({yrs:.1f} 年)")
    print(f"单位(1:1): 保证金 ¥{um:,.0f} | {yrs:.1f}年盈利 ¥{st['total_yuan']:,.0f} | "
          f"胜率 {st['win_rate']*100:.0f}% | {st['n_trades']} 笔")
    print("=" * 66)
    print(f"{'手数':>6}{'保证金':>10}{'杠杆':>6}{'总盈利':>12}{'简单年化':>9}{'最大回撤':>11}{'回撤%':>7}")
    best = None
    for lev in [0.2, 0.5, 1.0, 1.5, 2.0, 3.0]:
        units = max(1, int(cap * lev / um))
        pnl = ser * units
        eq = cap + pnl.cumsum()
        ddp = ((eq - eq.cummax()) / eq.cummax()).min()
        ann = pnl.sum() / cap / yrs
        print(f"{units:>6}{units*um:>10,.0f}{units*um/cap:>6.1f}x"
              f"{pnl.sum():>12,.0f}{ann*100:>8.0f}%{-(eq-eq.cummax()).min():>11,.0f}{-ddp*100:>6.0f}%")
        if abs(lev - 1.0) < 1e-9:
            best = (units, pnl, eq, ann, ddp)

    units, pnl, eq, ann, ddp = best
    print(f"\n推荐(满保证金~1x): {units} 手, 年化≈{ann*100:.0f}%, 最大回撤≈{-ddp*100:.0f}%, "
          f"年均 {st['n_trades']/yrs:.0f} 笔")
    print("⚠️ 单标的无分散, 历史低回撤不保证未来; 关系结构性断裂会放大亏损。")
    print("   务必: 杠杆从 1x 起步、守 4σ 止损、账户回撤 20% 熔断。")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(eq.index, eq.values, color="darkgreen", lw=2,
                label=f"{name} 1x: +¥{pnl.sum():,.0f} ({ann*100:.0f}%/yr), maxDD {-ddp*100:.0f}%")
        ax.axhline(cap, color="gray", ls="--", lw=0.8)
        ax.set_title(f"Single spread {name} - real money equity (capital ¥{cap:,.0f})")
        ax.set_ylabel("Account equity (CNY)"); ax.legend(); ax.grid(alpha=.3)
        plt.tight_layout()
        out = f"data/single_{a}_{b}_equity.png"
        plt.savefig(out, dpi=110)
        print(f"\n净值图已保存: {out}")
    except Exception as e:
        print(f"（绘图跳过: {e}）")


if __name__ == "__main__":
    main()
