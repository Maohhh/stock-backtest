#!/usr/bin/env python3
"""
买卖点「逻辑条件」穷举（指标参数全用默认，只改进场规则的组合）

指标默认：MACD(12,26,9)、LON(10,20)、MA20 离场连续 3 根。
把多头进场拆成三维并穷举（空头自动镜像）：
    ① LON 在 0 轴：上方 / 下方
    ② MACD：金叉 / 死叉
    ③ 交叉位置：0 轴上方 / 0 轴下方 / 不限
共 2×2×3 = 12 种买卖点定义。原版 = (LON上, 金叉, 0轴上)。

对每种规则在 44 品种上回测，给出等权组合指标、玻璃单品、并标注原版与“金叉在0轴下”变体。

用法： python lon_macd_rulesearch.py
"""

import os
import sys
import itertools

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lon_macd_backtest import DATA_DIR, ensure_data, load_symbol, COMMISSION  # noqa: E402
from src.indicators import macd as macd_ind, sma, lon as lon_ind  # noqa: E402
from src.backtest.futures_engine import run_backtest  # noqa: E402

OUT_DIR = "lon_macd_results"
ANNUAL = 252
WARMUP = 60
MA_PERIOD = 20
EXIT_BARS = 3


def precompute(df):
    m = macd_ind(df)
    dif = m["DIF"].to_numpy(); dea = m["DEA"].to_numpy()
    pg = np.r_[np.nan, dif[:-1]]; pe = np.r_[np.nan, dea[:-1]]
    golden = (dif > dea) & (pg <= pe)
    death = (dif < dea) & (pg >= pe)
    above = (dif > 0) & (dea > 0)
    below = (dif < 0) & (dea < 0)
    lv = lon_ind(df)["LON"].to_numpy()
    ma = sma(df, period=MA_PERIOD).to_numpy()
    close = df["close"].to_numpy()
    below_ma = close < ma
    above_ma = close > ma
    le = pd.Series(below_ma).rolling(EXIT_BARS).sum().to_numpy() == EXIT_BARS
    se = pd.Series(above_ma).rolling(EXIT_BARS).sum().to_numpy() == EXIT_BARS
    return dict(golden=golden, death=death, above=above, below=below,
                lon=lv, long_exit=np.nan_to_num(le).astype(bool),
                short_exit=np.nan_to_num(se).astype(bool))


def build_entries(pc, lon_dir, cross, axis):
    lon_pos = pc["lon"] > 0
    lon_neg = pc["lon"] < 0
    cross_map = {"金叉": pc["golden"], "死叉": pc["death"]}
    cross_map_flip = {"金叉": pc["death"], "死叉": pc["golden"]}
    axis_map = {"0轴上": pc["above"], "0轴下": pc["below"], "不限": np.ones_like(pc["above"], bool)}
    axis_flip = {"0轴上": pc["below"], "0轴下": pc["above"], "不限": np.ones_like(pc["above"], bool)}

    lc = lon_pos if lon_dir == "上" else lon_neg
    long_entry = lc & cross_map[cross] & axis_map[axis]
    # 镜像空头
    lc_s = lon_neg if lon_dir == "上" else lon_pos
    short_entry = lc_s & cross_map_flip[cross] & axis_flip[axis]
    return np.nan_to_num(long_entry).astype(bool), np.nan_to_num(short_entry).astype(bool)


def state_machine(le, se, lx, sx):
    n = len(le)
    st = np.zeros(n, np.int8); s = 0
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


def main():
    ensure_data()
    syms = sorted(f[:-4] for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
    rules = list(itertools.product(["上", "下"], ["金叉", "死叉"], ["0轴上", "0轴下", "不限"]))

    # 每个规则：收集各品种日收益 + 每品种夏普
    agg = {r: {"daily": {}, "sh": [], "ret": [], "ntr": []} for r in rules}
    fg = {}
    per_symbol_best = []
    for sym in syms:
        df = load_symbol(sym)
        pc = precompute(df)
        dates = pd.to_datetime(df["date"].values)
        close = df["close"].to_numpy()
        best = None
        for r in rules:
            le, se = build_entries(pc, *r)
            st = state_machine(le, se, pc["long_exit"], pc["short_exit"])
            sig = pd.DataFrame({"date": df["date"], "close": close, "state": st})
            res = run_backtest(sig, commission=COMMISSION)
            eq = res["equity_curve"]
            agg[r]["daily"][sym] = pd.Series(eq["equity"].pct_change().fillna(0.0).values, index=dates)
            agg[r]["sh"].append(res["sharpe"]); agg[r]["ret"].append(res["total_return"])
            agg[r]["ntr"].append(res["n_trades"])
            if sym == "FG0":
                fg[r] = (res["total_return"], res["sharpe"], res["n_trades"], res["win_rate"])
            if best is None or res["sharpe"] > best[1]:
                best = (r, res["sharpe"], res["total_return"])
        per_symbol_best.append({"品种": sym, "最优规则": "LON{}+{}+{}".format(*best[0]),
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
    for r in rules:
        cagr, sh, mdd = port(agg[r]["daily"])
        label = "LON{}+{}+{}".format(*r)
        tag = ""
        if r == ("上", "金叉", "0轴上"):
            tag = "★原版"
        elif r == ("上", "金叉", "0轴下"):
            tag = "☆你的变体"
        rows.append({
            "规则(多头)": label, "标注": tag,
            "组合年化%": round(cagr*100, 1), "组合夏普": round(sh, 2), "组合回撤%": round(mdd*100, 1),
            "品种夏普中位": round(np.median(agg[r]["sh"]), 2),
            "盈利品种%": round(np.mean(np.array(agg[r]["ret"]) > 0)*100, 0),
            "平均交易/品种": round(np.mean(agg[r]["ntr"]), 0),
            "玻璃收益%": round(fg[r][0]*100, 0) if r in fg else None,
        })
    rdf = pd.DataFrame(rows).sort_values("组合夏普", ascending=False).reset_index(drop=True)
    rdf.to_csv(os.path.join(OUT_DIR, "rule_search.csv"), index=False, encoding="utf-8-sig")
    pdf = pd.DataFrame(per_symbol_best)
    pdf.to_csv(os.path.join(OUT_DIR, "rule_per_symbol.csv"), index=False, encoding="utf-8-sig")

    R = ["# 买卖点逻辑条件穷举（指标参数默认）\n\n",
         "指标默认 MACD(12,26,9)/LON(10,20)/MA20；只改进场规则。多头 12 种，空头自动镜像；离场默认连续3根破MA20。\n\n",
         "## 全部 12 种买卖点（按组合夏普排序）\n",
         rdf.to_markdown(index=False), "\n\n"]
    best = rdf.iloc[0]
    orig = rdf[rdf["标注"] == "★原版"].iloc[0]
    var = rdf[rdf["标注"] == "☆你的变体"].iloc[0]
    R.append("## 关键观察\n")
    R.append(f"- **最优买卖点**：{best['规则(多头)']}（组合夏普 {best['组合夏普']}、年化 {best['组合年化%']}%、盈利品种 {best['盈利品种%']:.0f}%）。\n")
    R.append(f"- **原版**（LON上+金叉+0轴上）组合夏普 {orig['组合夏普']}、年化 {orig['组合年化%']}%。\n")
    R.append(f"- **你的变体**（LON上+金叉+0轴下）组合夏普 {var['组合夏普']}、年化 {var['组合年化%']}%、玻璃收益 {var['玻璃收益%']:.0f}%"
             f"（原版玻璃 {orig['玻璃收益%']:.0f}%）。\n")
    better = "更好" if var["组合夏普"] > orig["组合夏普"] else "更差"
    R.append(f"- 「金叉在0轴下方做多」相比原版「0轴上方」整体 **{better}**。\n")
    R.append("\n## 各品种最优买卖点\n")
    R.append(pdf.to_markdown(index=False))
    with open(os.path.join(OUT_DIR, "RULESEARCH.md"), "w", encoding="utf-8") as fh:
        fh.write("".join(R))
    print(rdf.to_string(index=False))
    print("\n" + "".join(R[6:11]))
    print(f"报告已保存: {OUT_DIR}/RULESEARCH.md")


if __name__ == "__main__":
    main()
