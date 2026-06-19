#!/usr/bin/env python3
"""
把「年化 3.4%」翻译成 10 万本金的真实盈亏：杠杆/暴露度的影响

要点：之前所有收益都是「满仓名义本金、不加杠杆」口径——
即假设你用全额合约价值的现金去持仓。但：
  1) 期货是保证金交易（约 10%），真实资金回报会被放大；
  2) 等权 26 个品种、且每个品种只有部分时间在场，组合「实际名义暴露」远低于 100%，
     大量资金闲置，所以名义口径年化看着低。
本脚本算出组合的实际暴露度，并给出不同杠杆下 10 万本金的年化收益与最大回撤。
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


def main():
    ensure_data()
    avail = set(f[:-4] for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
    syms = [s for s in INDUSTRIAL if s in avail]

    ret_df, expo = {}, {}
    for s in syms:
        df = load_symbol(s)
        r = run_backtest(atr_exit.generate_signals(df, k=2.0), commission=COMMISSION)
        eq = r["equity_curve"]
        ret_df[s] = pd.Series(eq["equity"].pct_change().fillna(0.0).values,
                              index=pd.to_datetime(eq["date"].values))
        expo[s] = eq["position"].abs().mean()  # 该品种在场时间占比

    mat = pd.DataFrame(ret_df).sort_index()
    port = mat.mean(axis=1, skipna=True).fillna(0.0)        # 等权组合日收益（满仓名义口径）
    # 实际名义暴露：每日 (1/N)·Σ|pos_i|，按当日有数据的品种数归一
    pos_mat = pd.DataFrame({s: ret_df[s].ne(0).astype(float) for s in syms})  # 占位
    gross = float(np.mean([expo[s] for s in syms]))         # 平均单品种在场占比≈组合平均毛暴露

    def stats(daily):
        eq = (1 + daily).cumprod()
        yrs = len(eq) / ANNUAL
        cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 and eq.iloc[-1] > 0 else 0
        mdd = (eq / eq.cummax() - 1).min()
        sh = daily.mean() / daily.std() * np.sqrt(ANNUAL) if daily.std() else 0
        return cagr, mdd, sh

    base_cagr, base_mdd, base_sh = stats(port)

    print("# 10 万本金能赚多少：杠杆/暴露度翻译\n")
    print(f"工业品篮子（{len(syms)} 个，ATR k=2，等权）满仓名义口径：")
    print(f"  年化 {base_cagr*100:.1f}%  最大回撤 {base_mdd*100:.1f}%  夏普 {base_sh:.2f}")
    print(f"  组合平均「实际名义暴露」≈ {gross*100:.0f}%（即平均只有这么多资金真正在市场里，其余闲置）\n")

    print("## 不同杠杆下，10 万本金的年化收益与最大回撤")
    print("（杠杆=名义/本金。期货保证金约10%，所以 5~10x 在保证金上是可行的，但回撤同步放大）\n")
    print(f"{'杠杆':>5} | {'年化收益':>10} | {'≈每年(元)':>12} | {'最大回撤':>9} | {'≈回撤(元)':>11}")
    print("-" * 60)
    for lev in [1, 2, 3, 5, 8]:
        d = port * lev
        # 杠杆下单日收益不可能 < -100%，做个保护（实际会触发追保/强平）
        cagr, mdd, _ = stats(d.clip(lower=-0.99))
        print(f"{lev:>4}x | {cagr*100:>8.1f}% | {cagr*CAPITAL:>11,.0f} | {mdd*100:>7.1f}% | {mdd*CAPITAL:>10,.0f}")

    print("\n## 对照：若只重仓单个最强品种（满仓名义、不加杠杆）")
    for s in ["CU0", "FG0", "AU0"]:
        cagr, mdd, sh = stats(ret_df[s])
        print(f"  {s}: 年化 {cagr*100:5.1f}%  回撤 {mdd*100:6.1f}%  夏普 {sh:.2f}  "
              f"-> 10万一年≈{cagr*CAPITAL:,.0f}元")

    print("\n要点：")
    print("- 3.4% 是「满仓名义、不加杠杆」且资金大半闲置（暴露≈{:.0f}%）的结果，不是账户真实回报。".format(gross*100))
    print("- 期货保证金~10%，把暴露提到接近满仓（3~5x 杠杆）后，10万一年量级在 1万~1.7万，但回撤也到 -25%~-40%。")
    print("- 唯一不随杠杆变的是夏普 {:.2f}——这才是策略质量。杠杆只是同时放大收益和亏损/爆仓风险。".format(base_sh))


if __name__ == "__main__":
    main()
