#!/usr/bin/env python3
"""
最优策略 + 10万本金 + 50%资金使用率(≈5x杠杆)：一年赚多少

最优策略：买点原版(LON>0+金叉+0轴上) + 卖点 连续3根破MA10（全局穷举+样本外验证的稳健最优），
做工业品篮子(26个)等权。50% 资金当保证金、保证金率~10% => ≈5x 名义杠杆。
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lon_macd_backtest import DATA_DIR, ensure_data, load_symbol, COMMISSION  # noqa: E402
from src.strategies.lon_macd_strategy import generate_signals  # noqa: E402
from src.backtest.futures_engine import run_backtest  # noqa: E402

ANNUAL = 252
CAPITAL = 100_000
INDUSTRIAL = ["AU0", "AG0", "AL0", "CU0", "NI0", "PB0", "SN0", "ZN0",
              "RB0", "HC0", "I0", "JM0", "SF0", "SM0", "SS0",
              "TA0", "MA0", "PP0", "L0", "V0", "EG0", "FU0", "BU0", "SC0", "RU0", "FG0"]


def main():
    ensure_data()
    avail = set(f[:-4] for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
    syms = [s for s in INDUSTRIAL if s in avail]

    ret, expo = {}, []
    for s in syms:
        df = load_symbol(s)
        # 最优策略：买点原版 + 卖点连续3根破MA10
        r = run_backtest(generate_signals(df, ma_period=10, exit_bars=3, require_zero_axis=True),
                         commission=COMMISSION)
        eq = r["equity_curve"]
        ret[s] = pd.Series(eq["equity"].pct_change().fillna(0.0).values,
                           index=pd.to_datetime(eq["date"].values))
        expo.append(eq["position"].abs().mean())

    mat = pd.DataFrame(ret).sort_index()
    port = mat.mean(axis=1, skipna=True).fillna(0.0)
    avg_expo = float(np.mean(expo))

    def stats(daily):
        eq = (1 + daily.clip(lower=-0.99)).cumprod()
        yrs = len(eq) / ANNUAL
        cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 and eq.iloc[-1] > 0 else -1
        mdd = (eq / eq.cummax() - 1).min()
        sh = daily.mean() / daily.std() * np.sqrt(ANNUAL) if daily.std() else 0
        return cagr, mdd, sh, daily.min()

    cagr, mdd, sh, wd = stats(port)
    print("# 最优策略 + 10万本金 + 50%资金使用率，一年赚多少\n")
    print("策略：买点 LON>0+金叉+0轴上；卖点 连续3根破MA10；工业品篮子26个等权。")
    print(f"满仓名义口径：年化 {cagr*100:.1f}%  最大回撤 {mdd*100:.1f}%  夏普 {sh:.2f}")
    print(f"组合平均实际名义暴露 ≈ {avg_expo*100:.0f}%（大部分时间空仓，资金闲置）\n")

    print("## 不同资金使用率(杠杆)下，10万本金一年的收益与回撤")
    print(f"{'资金使用率':>8} {'杠杆':>5} | {'年化':>8} | {'10万一年≈':>11} | {'最大回撤':>9} | {'最惨单日':>9}")
    print("-" * 66)
    for util, lev in [(0.20, 2.0), (0.30, 3.0), (0.50, 5.0), (0.75, 7.5)]:
        c, m, _, w = stats(port * lev)
        mark = "  <= 你问的" if abs(util - 0.50) < 1e-9 else ""
        print(f"{util*100:>6.0f}% {lev:>4.1f}x | {c*100:>6.1f}% | {c*CAPITAL:>10,.0f} | "
              f"{m*100:>7.1f}% | {w*100:>7.1f}%{mark}")

    c5, m5, _, w5 = stats(port * 5.0)
    print("\n要点（50% 资金使用率 ≈ 5x 杠杆）：")
    print(f"- 年化约 {c5*100:.0f}%，**10万一年≈{c5*CAPITAL:,.0f}元**；但最大回撤 {m5*100:.0f}%"
          f"（10万一度浮亏到 {CAPITAL*(1+m5):,.0f}元），最惨单日 {w5*100:.0f}%。")
    print(f"- 这是工业品篮子26个分散后的结果；若集中单品种 5x，遇到换月跳点/极端日仍可能强平。")
    print(f"- 该策略平均只有 ~{avg_expo*100:.0f}% 暴露，所谓 50% 使用率是“信号多时峰值占用”，平均占用远低于此。")
    print(f"- 现实预期：50% 仓位偏激进，{m5*100:.0f}% 回撤多数人扛不住；更稳是 20-30%（2-3x），一年 ~{stats(port*2.5)[0]*CAPITAL:,.0f}元。")


if __name__ == "__main__":
    main()
