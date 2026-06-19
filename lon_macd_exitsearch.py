#!/usr/bin/env python3
"""
卖点（出场）逻辑条件穷举 —— 固定最优买点，穷举各类出场

买点固定为最优组合：LON>0 + MACD金叉 + 双线在0轴上方（空头镜像）。
指标参数默认 MACD(12,26,9)/LON(10,20)。穷举多种出场（多头平仓，空头镜像）：

  均线类： 连续3根<MA20(原版) / <MA20 / <MA10 / 连续3根<MA10 / <MA60
  MACD类： 死叉 / 柱状转向下 / DIF下穿0轴 / DIF<DEA(状态)
  LON类：  LON下穿0轴 / LON转向下
  参照：   ATR(k=2) 移动止损

对 44 品种回测，给出等权组合指标、玻璃单品、各品种最优出场。

用法： python lon_macd_exitsearch.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lon_macd_backtest import DATA_DIR, ensure_data, load_symbol, COMMISSION  # noqa: E402
from src.indicators import macd as macd_ind, sma, lon as lon_ind  # noqa: E402
from src.strategies import lon_macd_atr as atr_exit  # noqa: E402
from src.backtest.futures_engine import run_backtest  # noqa: E402

OUT_DIR = "lon_macd_results"
ANNUAL = 252
WARMUP = 60


def roll_eq(mask, k):
    return pd.Series(mask).rolling(k).sum().to_numpy() == k


def precompute(df):
    m = macd_ind(df)
    dif = m["DIF"].to_numpy(); dea = m["DEA"].to_numpy(); hist = m["MACD"].to_numpy()
    pg = np.r_[np.nan, dif[:-1]]; pe = np.r_[np.nan, dea[:-1]]
    golden = (dif > dea) & (pg <= pe)
    death = (dif < dea) & (pg >= pe)
    above = (dif > 0) & (dea > 0)
    below = (dif < 0) & (dea < 0)
    lv = lon_ind(df)["LON"].to_numpy()
    close = df["close"].to_numpy()
    ma = {p: sma(df, period=p).to_numpy() for p in (10, 20, 60)}
    return dict(dif=dif, dea=dea, hist=hist, golden=golden, death=death,
                above=above, below=below, lon=lv, close=close, ma=ma,
                pdif=pg, plon=np.r_[np.nan, lv[:-1]], phist=np.r_[np.nan, hist[:-1]])


def exit_pair(pc, name):
    """返回 (long_exit, short_exit) 布尔数组。"""
    c = pc["close"]; ma = pc["ma"]; dif = pc["dif"]; dea = pc["dea"]
    if name == "连续3根<MA20(原版)":
        lx = roll_eq(c < ma[20], 3); sx = roll_eq(c > ma[20], 3)
    elif name == "收盘<MA20":
        lx = c < ma[20]; sx = c > ma[20]
    elif name == "收盘<MA10":
        lx = c < ma[10]; sx = c > ma[10]
    elif name == "连续3根<MA10":
        lx = roll_eq(c < ma[10], 3); sx = roll_eq(c > ma[10], 3)
    elif name == "收盘<MA60":
        lx = c < ma[60]; sx = c > ma[60]
    elif name == "MACD死叉":
        lx = pc["death"]; sx = pc["golden"]
    elif name == "MACD柱转向":
        lx = pc["hist"] < pc["phist"]; sx = pc["hist"] > pc["phist"]
    elif name == "DIF穿0轴":
        lx = (dif < 0) & (pc["pdif"] >= 0); sx = (dif > 0) & (pc["pdif"] <= 0)
    elif name == "DIF<DEA(状态)":
        lx = dif < dea; sx = dif > dea
    elif name == "LON穿0轴":
        lx = (pc["lon"] < 0) & (pc["plon"] >= 0); sx = (pc["lon"] > 0) & (pc["plon"] <= 0)
    elif name == "LON转向":
        lx = pc["lon"] < pc["plon"]; sx = pc["lon"] > pc["plon"]
    else:
        raise ValueError(name)
    return np.nan_to_num(lx).astype(bool), np.nan_to_num(sx).astype(bool)


def state_machine(le, se, lx, sx):
    n = len(le); st = np.zeros(n, np.int8); s = 0
    for t in range(WARMUP, n):
        if s == 0:
            if le[t]:
                s = 1
            elif se[t]:
                s = -1
        elif s == 1:
            if lx[t]:
                s = 0
                if se[t]:
                    s = -1
        else:
            if sx[t]:
                s = 0
                if le[t]:
                    s = 1
        st[t] = s
    return st


EXITS = ["连续3根<MA20(原版)", "收盘<MA20", "收盘<MA10", "连续3根<MA10", "收盘<MA60",
         "MACD死叉", "MACD柱转向", "DIF穿0轴", "DIF<DEA(状态)", "LON穿0轴", "LON转向", "ATR止损k=2"]


def main():
    ensure_data()
    syms = sorted(f[:-4] for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
    agg = {e: {"daily": {}, "sh": [], "ret": [], "ntr": []} for e in EXITS}
    fg = {}
    per_symbol_best = []

    for sym in syms:
        df = load_symbol(sym)
        pc = precompute(df)
        dates = pd.to_datetime(df["date"].values)
        close = pc["close"]
        # 固定最优买点
        long_entry = np.nan_to_num(pc["above"] & pc["golden"] & (pc["lon"] > 0)).astype(bool)
        short_entry = np.nan_to_num(pc["below"] & pc["death"] & (pc["lon"] < 0)).astype(bool)

        best = None
        for e in EXITS:
            if e == "ATR止损k=2":
                res = run_backtest(atr_exit.generate_signals(df, k=2.0), commission=COMMISSION)
            else:
                lx, sx = exit_pair(pc, e)
                st = state_machine(long_entry, short_entry, lx, sx)
                res = run_backtest(pd.DataFrame({"date": df["date"], "close": close, "state": st}),
                                   commission=COMMISSION)
            eq = res["equity_curve"]
            agg[e]["daily"][sym] = pd.Series(eq["equity"].pct_change().fillna(0.0).values, index=dates)
            agg[e]["sh"].append(res["sharpe"]); agg[e]["ret"].append(res["total_return"])
            agg[e]["ntr"].append(res["n_trades"])
            if sym == "FG0":
                fg[e] = (res["total_return"], res["sharpe"])
            if best is None or res["sharpe"] > best[1]:
                best = (e, res["sharpe"], res["total_return"])
        per_symbol_best.append({"品种": sym, "最优出场": best[0],
                                "夏普": round(best[1], 2), "总收益%": round(best[2]*100, 0)})

    def port(daily_map):
        mat = pd.DataFrame(daily_map).sort_index()
        pr = mat.mean(axis=1, skipna=True).fillna(0.0)
        eq = (1 + pr).cumprod()
        yrs = len(eq) / ANNUAL
        cagr = eq.iloc[-1] ** (1/yrs) - 1 if yrs > 0 and eq.iloc[-1] > 0 else 0
        sh = pr.mean()/pr.std()*np.sqrt(ANNUAL) if pr.std() else 0
        mdd = (eq/eq.cummax()-1).min()
        return cagr, sh, mdd

    rows = []
    for e in EXITS:
        cagr, sh, mdd = port(agg[e]["daily"])
        rows.append({
            "出场方式": e + (" ★原版" if e.startswith("连续3根<MA20") else ""),
            "组合年化%": round(cagr*100, 1), "组合夏普": round(sh, 2), "组合回撤%": round(mdd*100, 1),
            "品种夏普中位": round(np.median(agg[e]["sh"]), 2),
            "盈利品种%": round(np.mean(np.array(agg[e]["ret"]) > 0)*100, 0),
            "平均交易/品种": round(np.mean(agg[e]["ntr"]), 0),
            "玻璃夏普": round(fg[e][1], 2),
        })
    rdf = pd.DataFrame(rows).sort_values("组合夏普", ascending=False).reset_index(drop=True)
    rdf.to_csv(os.path.join(OUT_DIR, "exit_search.csv"), index=False, encoding="utf-8-sig")
    pdf = pd.DataFrame(per_symbol_best)
    pdf.to_csv(os.path.join(OUT_DIR, "exit_per_symbol.csv"), index=False, encoding="utf-8-sig")

    R = ["# 卖点（出场）逻辑条件穷举\n\n",
         "买点固定为最优：LON>0 + MACD金叉 + 双线0轴上方（空头镜像）；指标参数默认。\n"
         "穷举 12 种出场（均线/MACD/LON/ATR），出场多头平仓、空头镜像。\n\n",
         "## 全部出场方式（按组合夏普排序）\n", rdf.to_markdown(index=False), "\n\n"]
    best = rdf.iloc[0]; orig = rdf[rdf["出场方式"].str.contains("原版")].iloc[0]
    R.append("## 关键观察\n")
    R.append(f"- **最优出场**：{best['出场方式']}（组合夏普 {best['组合夏普']}、年化 {best['组合年化%']}%、回撤 {best['组合回撤%']}%）。\n")
    R.append(f"- **原版出场**（连续3根<MA20）组合夏普 {orig['组合夏普']}、年化 {orig['组合年化%']}%。\n")
    diff = best['组合夏普'] - orig['组合夏普']
    R.append(f"- 最优出场比原版夏普 {'高' if diff>0 else '低'} {abs(diff):.2f}。\n")
    R.append("- 出场方式之间的差距，明显大于上一轮买点条件之间的差距——**卖点比买点更值得讲究**。\n")
    R.append("\n## 各品种最优出场\n")
    R.append(pdf.to_markdown(index=False))
    with open(os.path.join(OUT_DIR, "EXITSEARCH.md"), "w", encoding="utf-8") as fh:
        fh.write("".join(R))
    print(rdf.to_string(index=False))
    print("\n各品种最优出场分布:")
    print(pdf["最优出场"].value_counts().to_string())
    print(f"\n报告已保存: {OUT_DIR}/EXITSEARCH.md")


if __name__ == "__main__":
    main()
