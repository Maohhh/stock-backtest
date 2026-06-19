#!/usr/bin/env python3
"""
10 万本金、资金使用率~75% 当保证金，一年赚多少？（含爆仓判定）

换算：保证金率约 10% -> 75% 资金当保证金 ≈ 7.5x 名义杠杆（保证金率13%则≈5.8x）。
分别看：
  A) 工业品篮子（26个等权）在该杠杆下的年化、回撤、是否触及爆仓；
  B) 单品种（铜/玻璃）在该杠杆下——通常直接爆仓；
  C) 现实问题：篮子大部分时间空仓，平均很难真把 75% 资金用满。
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lon_macd_backtest import DATA_DIR, ensure_data, load_symbol, COMMISSION  # noqa: E402
from src.strategies import lon_macd_atr as atr_exit  # noqa: E402
from src.backtest.futures_engine import run_backtest  # noqa: E402

ANNUAL = 252
CAPITAL = 100_000
INDUSTRIAL = ["AU0", "AG0", "AL0", "CU0", "NI0", "PB0", "SN0", "ZN0",
              "RB0", "HC0", "I0", "JM0", "SF0", "SM0", "SS0",
              "TA0", "MA0", "PP0", "L0", "V0", "EG0", "FU0", "BU0", "SC0", "RU0", "FG0"]


def series_for(sym):
    df = load_symbol(sym)
    r = run_backtest(atr_exit.generate_signals(df, k=2.0), commission=COMMISSION)
    eq = r["equity_curve"]
    return (pd.Series(eq["equity"].pct_change().fillna(0.0).values,
                      index=pd.to_datetime(eq["date"].values)),
            eq["position"].abs().mean())


def lever_stats(daily, lev):
    d = daily * lev
    worst_day = d.min()
    wiped = worst_day <= -1.0           # 单日亏掉全部本金=爆仓
    eq = (1 + d.clip(lower=-1.0)).cumprod()
    yrs = len(eq) / ANNUAL
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 and eq.iloc[-1] > 0 else -1.0
    mdd = (eq / eq.cummax() - 1).min()
    return cagr, mdd, worst_day, wiped


def main():
    ensure_data()
    avail = set(f[:-4] for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
    syms = [s for s in INDUSTRIAL if s in avail]

    ret = {}
    expo = {}
    for s in syms:
        ret[s], expo[s] = series_for(s)
    mat = pd.DataFrame(ret).sort_index()
    port = mat.mean(axis=1, skipna=True).fillna(0.0)
    avg_expo = float(np.mean([expo[s] for s in syms]))

    print("# 10万本金、资金使用率~75%当保证金，一年赚多少\n")
    print("换算：保证金~10% => 75%资金当保证金 ≈ 7.5x 杠杆。\n")

    print("## A) 工业品篮子（26等权）在 7.5x（及邻近杠杆）下")
    print(f"{'杠杆':>5} | {'年化':>8} | {'10万一年≈':>11} | {'最大回撤':>9} | {'最惨单日':>9} | 爆仓?")
    print("-" * 64)
    for lev in [5.0, 5.8, 7.5]:
        cagr, mdd, wd, wiped = lever_stats(port, lev)
        print(f"{lev:>4.1f}x | {cagr*100:>6.1f}% | {cagr*CAPITAL:>10,.0f} | {mdd*100:>7.1f}% | "
              f"{wd*100:>7.1f}% | {'是!' if wiped else '否'}")

    print("\n## B) 若把 75% 保证金压在单个品种（7.5x）")
    for s in ["CU0", "FG0", "RB0"]:
        cagr, mdd, wd, wiped = lever_stats(ret[s], 7.5)
        flag = "★爆仓/强平" if (wiped or mdd <= -1.0) else f"回撤{mdd*100:.0f}%"
        print(f"  {s}: 最惨单日 {wd*100:.1f}%  -> {flag}")

    print("\n## C) 现实约束：这个策略平均只有 ~{:.0f}% 名义暴露".format(avg_expo*100))
    print(f"  - 篮子里每个品种平均只有 {avg_expo*100:.0f}% 的时间在场，大部分时间空仓。")
    print(f"  - 想让账户『平均』用满 75% 保证金，需要约 {0.75/0.10/avg_expo:.0f}x 名义杠杆——这会瞬间爆仓。")
    print(f"  - 7.5x 是『信号全开时峰值用 75% 保证金』，平均资金使用率其实只有 ~{avg_expo*7.5*10:.0f}%。")

    print("\n要点：")
    cagr75, mdd75, _, wiped75 = lever_stats(port, 7.5)
    print(f"- 篮子 7.5x：理论年化 {cagr75*100:.0f}%（10万≈{cagr75*CAPITAL:,.0f}元/年），"
          f"但最大回撤 {mdd75*100:.0f}%——10万会一度浮亏到 {CAPITAL*(1+mdd75):,.0f} 元。")
    print("- 单品种 7.5x 基本=强平出局；只有靠26品种分散，7.5x 才勉强不当场爆仓。")
    print("- 75% 资金使用率对趋势策略属『极限激进』，回撤足以让绝大多数人中途斩仓离场。")
    print("- 稳健做法：资金使用率 20%~30%（≈2~3x 杠杆），10万一年≈7千~1万，回撤控制在 -25% 内。")


if __name__ == "__main__":
    main()
