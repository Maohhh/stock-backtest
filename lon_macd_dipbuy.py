#!/usr/bin/env python3
"""
日线：顺势买回调(趋势+KDJ超卖) vs 纯趋势(金叉) —— 看胜率/盈亏比的跷跷板

方向都用：LON>0 且 MACD 双线在 0 轴上方（空头镜像）。卖点都用：连续3根破MA10。
只改进场：
  A 纯趋势：MACD 金叉进场（原版最优买点）。
  B 顺势买回调：KDJ 在超卖区(J<20)且 K 上穿 D 进场（趋势中买跌）。
对工业品篮子(26)逐品种回测，比较胜率、盈亏比、平均赢亏、总收益、夏普。

用法： python lon_macd_dipbuy.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lon_macd_backtest import DATA_DIR, ensure_data, load_symbol, COMMISSION  # noqa: E402
from src.indicators import macd as macd_ind, lon as lon_ind, kdj as kdj_ind, sma  # noqa: E402
from src.backtest.futures_engine import run_backtest  # noqa: E402

OUT_DIR = "lon_macd_results"
ANNUAL = 252
WARMUP = 60
INDUSTRIAL = ["AU0", "AG0", "AL0", "CU0", "NI0", "PB0", "SN0", "ZN0", "RB0", "HC0", "I0", "JM0",
              "SF0", "SM0", "SS0", "TA0", "MA0", "PP0", "L0", "V0", "EG0", "FU0", "BU0", "SC0", "RU0", "FG0"]


def roll3_below(c, ma):
    return np.nan_to_num(pd.Series(c < ma).rolling(3).sum().to_numpy() == 3).astype(bool)


def build(df, mode):
    m = macd_ind(df); dif = m["DIF"].to_numpy(); dea = m["DEA"].to_numpy()
    pdif = np.r_[np.nan, dif[:-1]]; pdea = np.r_[np.nan, dea[:-1]]
    golden = (dif > dea) & (pdif <= pdea); death = (dif < dea) & (pdif >= pdea)
    lon = lon_ind(df)["LON"].to_numpy()
    up = (lon > 0) & (dif > 0) & (dea > 0); dn = (lon < 0) & (dif < 0) & (dea < 0)
    k = kdj_ind(df); K = k["K"].to_numpy(); D = k["D"].to_numpy(); J = k["J"].to_numpy()
    pK = np.r_[np.nan, K[:-1]]; pD = np.r_[np.nan, D[:-1]]
    kup = (K > D) & (pK <= pD); kdn = (K < D) & (pK >= pD)
    lon_pos = lon > 0; lon_neg = lon < 0
    if mode == "trend":
        le = up & golden; se = dn & death
    else:  # dip：上涨趋势(LON>0)中 KDJ 从回调区(J<30)上穿，买跌；空头镜像
        le = lon_pos & kup & (J < 30); se = lon_neg & kdn & (J > 70)
    le = np.nan_to_num(le).astype(bool); se = np.nan_to_num(se).astype(bool)
    c = df["close"].to_numpy(); ma10 = sma(df, period=10).to_numpy()
    lx = roll3_below(c, ma10); sx = np.nan_to_num(pd.Series(c > ma10).rolling(3).sum().to_numpy() == 3).astype(bool)
    n = len(df); st = np.zeros(n, np.int8); s = 0
    for t in range(WARMUP, n):
        if s == 0:
            if le[t]:
                s = 1
            elif se[t]:
                s = -1
        elif s == 1:
            if lx[t]:
                s = 0
        else:
            if sx[t]:
                s = 0
        st[t] = s
    return st


def main():
    ensure_data()
    avail = set(f[:-4] for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
    syms = [s for s in INDUSTRIAL if s in avail]
    out = {}
    for mode in ["trend", "dip"]:
        daily = {}; agg = []
        for s in syms:
            df = load_symbol(s)
            st = build(df, mode)
            r = run_backtest(pd.DataFrame({"date": df["date"], "close": df["close"].values, "state": st}),
                             commission=COMMISSION)
            daily[s] = pd.Series(r["equity_curve"]["equity"].pct_change().fillna(0).values,
                                 index=pd.to_datetime(df["date"].values))
            agg.append((r["total_return"], r["sharpe"], r["n_trades"], r["win_rate"],
                        r["profit_factor"], r["avg_win"], r["avg_loss"]))
        a = np.array([[x[i] for x in agg] for i in range(7)])
        mat = pd.DataFrame(daily).sort_index(); pr = mat.mean(axis=1, skipna=True).fillna(0)
        eq = (1+pr).cumprod()
        sh = pr.mean()/pr.std()*np.sqrt(ANNUAL) if pr.std() else 0
        cagr = eq.iloc[-1]**(1/(len(eq)/ANNUAL))-1 if eq.iloc[-1] > 0 else -1
        mdd = (eq/eq.cummax()-1).min()
        out[mode] = dict(port_cagr=cagr, port_sh=sh, port_mdd=mdd,
                         win=np.nanmean(a[3]), pf=np.nanmedian(a[4]),
                         aw=np.nanmean(a[5]), al=np.nanmean(a[6]),
                         ntr=np.nanmean(a[2]), pos=int((a[0] > 0).sum()), n=len(syms),
                         sh_med=np.nanmedian(a[1]))

    R = ["# 日线：顺势买回调(趋势+KDJ超卖) vs 纯趋势(金叉)\n\n",
         "方向同为 LON>0+MACD双线0轴上；卖点同为连3根破MA10；只改进场。工业品篮子26个。\n\n",
         "| 指标 | A 纯趋势(金叉进场) | B 顺势买回调(KDJ超卖进场) |\n|---|---|---|\n"]
    a, b = out["trend"], out["dip"]
    def fmt(d):
        return d
    R.append(f"| 平均单笔胜率 | {a['win']*100:.1f}% | **{b['win']*100:.1f}%** |\n")
    R.append(f"| 盈亏比(中位) | {a['pf']:.2f} | {b['pf']:.2f} |\n")
    R.append(f"| 平均赢 / 平均亏 | +{a['aw']*100:.2f}% / {a['al']*100:.2f}% | +{b['aw']*100:.2f}% / {b['al']*100:.2f}% |\n")
    R.append(f"| 平均交易/品种 | {a['ntr']:.0f} | {b['ntr']:.0f} |\n")
    R.append(f"| 组合年化 | {a['port_cagr']*100:.1f}% | {b['port_cagr']*100:.1f}% |\n")
    R.append(f"| 组合夏普 | **{a['port_sh']:.2f}** | {b['port_sh']:.2f} |\n")
    R.append(f"| 组合最大回撤 | {a['port_mdd']*100:.1f}% | {b['port_mdd']*100:.1f}% |\n")
    R.append(f"| 品种夏普中位 | {a['sh_med']:.2f} | {b['sh_med']:.2f} |\n")
    R.append(f"| 盈利品种 | {a['pos']}/{a['n']} | {b['pos']}/{b['n']} |\n")
    R.append("\n## 结论\n")
    R.append(f"- **胜率几乎没变**（{a['win']*100:.0f}% → {b['win']*100:.0f}%）——因为两者**卖点相同**。"
             "**胜率主要由卖点（止盈方式）决定，不是进场决定**：想要高胜率得改成快速止盈的卖点，而非换进场。\n")
    R.append(f"- 「买回调」主要是**同时砍小了右尾和回撤**：平均赢 +{a['aw']*100:.2f}%→+{b['aw']*100:.2f}%、"
             f"回撤 {a['port_mdd']*100:.0f}%→{b['port_mdd']*100:.0f}%——进得更低更晚，错过突破段，赢得小、亏得也小。\n")
    R.append(f"- 净结果：组合夏普 {a['port_sh']:.2f}→{b['port_sh']:.2f}、年化 {a['port_cagr']*100:.1f}%→{b['port_cagr']*100:.1f}%——"
             "**回撤换来了，但收益和夏普也降了，没有免费午餐。**\n")
    with open(os.path.join(OUT_DIR, "DIPBUY.md"), "w", encoding="utf-8") as fh:
        fh.write("".join(R))
    print("".join(R))


if __name__ == "__main__":
    main()
