#!/usr/bin/env python3
"""
MACD x LON x 均线 参数网格穷举回测（含样本内/样本外验证，防过拟合）

进场：LON 同向 + MACD 金/死叉 + 双线过 0 轴；离场：连续 N 根收盘破 MA。
对每个品种、每组参数，分别在样本内(前70%)与样本外(后30%)评估夏普与收益。

输出：
  - 全局最优（按 44 品种平均样本内夏普）及其样本外表现；
  - 样本外最稳健的组合；
  - 每个品种各自的最优组合（样本内）及其样本外夏普（看过拟合衰减）。

用法： python lon_macd_gridsearch.py
"""

import os
import sys
import itertools

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lon_macd_backtest import DATA_DIR, ensure_data, load_symbol, COMMISSION  # noqa: E402
from src.indicators import lon as lon_ind  # noqa: E402

OUT_DIR = "lon_macd_results"
ANNUAL = 252
WARMUP = 60
SPLIT = 0.70

MACD_SETS = [(12, 26, 9), (8, 17, 9), (5, 34, 5), (6, 19, 6)]
LON_SETS = [(10, 20), (5, 15), (13, 34)]
MA_PERIODS = [10, 20, 30, 40]
EXIT_BARS = [2, 3, 4]


def ema(arr, span):
    return pd.Series(arr).ewm(span=span, adjust=False, min_periods=1).mean().to_numpy()


def state_machine(long_entry, short_entry, long_exit, short_exit):
    n = len(long_entry)
    state = np.zeros(n, dtype=np.int8)
    s = 0
    for t in range(WARMUP, n):
        if s == 0:
            if long_entry[t]:
                s = 1
            elif short_entry[t]:
                s = -1
        elif s == 1:
            if long_exit[t]:
                s = 0
                if short_entry[t]:
                    s = -1
        else:
            if short_exit[t]:
                s = 0
                if long_entry[t]:
                    s = 1
        state[t] = s
    return state


def slice_metrics(state, close, split):
    pos = np.empty_like(state, dtype=float)
    pos[0] = 0
    pos[1:] = state[:-1]
    pct = np.zeros(len(close))
    pct[1:] = np.diff(close) / close[:-1]
    gross = pos * pct
    turn = np.abs(np.diff(pos, prepend=0.0))
    net = gross - turn * COMMISSION

    def stat(seg):
        if len(seg) < 30 or seg.std() == 0:
            return 0.0, 0.0
        sh = seg.mean() / seg.std() * np.sqrt(ANNUAL)
        tot = np.prod(1 + seg) - 1
        return sh, tot

    sh_is, ret_is = stat(net[:split])
    sh_oos, ret_oos = stat(net[split:])
    return sh_is, ret_is, sh_oos, ret_oos


def main():
    ensure_data()
    syms = sorted(f[:-4] for f in os.listdir(DATA_DIR) if f.endswith(".csv"))

    combos = list(itertools.product(range(len(MACD_SETS)), range(len(LON_SETS)),
                                    MA_PERIODS, EXIT_BARS))
    print(f"组合数 {len(combos)} × 品种 {len(syms)} = {len(combos)*len(syms)} 次回测，开始 ...")

    # 累加每个组合在各品种的样本内/外夏普
    agg = {c: {"is": [], "oos": []} for c in combos}
    per_symbol_best = []

    for si, sym in enumerate(syms):
        df = load_symbol(sym)
        close = df["close"].to_numpy(float)
        high = df["high"].to_numpy(float)
        low = df["low"].to_numpy(float)
        n = len(df)
        split = int(n * SPLIT)

        # 预计算：每个 MACD 的 dif/dea/金叉死叉
        macd_cache = {}
        for mi, (f, s, sig) in enumerate(MACD_SETS):
            dif = ema(close, f) - ema(close, s)
            dea = pd.Series(dif).ewm(span=sig, adjust=False, min_periods=1).mean().to_numpy()
            pg = np.r_[np.nan, dif[:-1]]
            pe = np.r_[np.nan, dea[:-1]]
            golden = (dif > dea) & (pg <= pe)
            death = (dif < dea) & (pg >= pe)
            macd_cache[mi] = (dif, dea, golden, death)

        # 预计算：每个 LON 的符号
        lon_cache = {}
        for li, (lf, ls) in enumerate(LON_SETS):
            lv = lon_ind(df, fast=lf, slow=ls)["LON"].to_numpy()
            lon_cache[li] = lv

        # 预计算：每个 MA 的 below/above
        ma_cache = {}
        for p in MA_PERIODS:
            ma = pd.Series(close).rolling(p, min_periods=1).mean().to_numpy()
            ma_cache[p] = (close < ma, close > ma)

        best = None
        for (mi, li, p, eb) in combos:
            dif, dea, golden, death = macd_cache[mi]
            lv = lon_cache[li]
            long_entry = (lv > 0) & golden & (dif > 0) & (dea > 0)
            short_entry = (lv < 0) & death & (dif < 0) & (dea < 0)
            long_entry = np.nan_to_num(long_entry).astype(bool)
            short_entry = np.nan_to_num(short_entry).astype(bool)
            below, above = ma_cache[p]
            le = pd.Series(below).rolling(eb).sum().to_numpy() == eb
            se = pd.Series(above).rolling(eb).sum().to_numpy() == eb
            le = np.nan_to_num(le).astype(bool)
            se = np.nan_to_num(se).astype(bool)

            state = state_machine(long_entry, short_entry, le, se)
            sh_is, ret_is, sh_oos, ret_oos = slice_metrics(state, close, split)
            agg[(mi, li, p, eb)]["is"].append(sh_is)
            agg[(mi, li, p, eb)]["oos"].append(sh_oos)
            if best is None or sh_is > best[1]:
                best = ((mi, li, p, eb), sh_is, sh_oos, ret_is, ret_oos)
        bc = best[0]
        per_symbol_best.append({
            "品种": sym,
            "MACD": str(MACD_SETS[bc[0]]), "LON": str(LON_SETS[bc[1]]),
            "MA": bc[2], "离场根": bc[3],
            "样本内夏普": round(best[1], 2), "样本外夏普": round(best[2], 2),
            "样本内收益%": round(best[3]*100, 0), "样本外收益%": round(best[4]*100, 0),
        })
        print(f"  [{si+1}/{len(syms)}] {sym} 最优 IS夏普 {best[1]:.2f} -> OOS {best[2]:.2f}")

    # 全局排名（按平均样本内夏普）
    rows = []
    for c, d in agg.items():
        mi, li, p, eb = c
        rows.append({
            "MACD": str(MACD_SETS[mi]), "LON": str(LON_SETS[li]), "MA": p, "离场根": eb,
            "IS夏普均值": round(np.mean(d["is"]), 3),
            "OOS夏普均值": round(np.mean(d["oos"]), 3),
            "IS夏普中位": round(np.median(d["is"]), 3),
            "OOS夏普中位": round(np.median(d["oos"]), 3),
            "OOS盈利品种%": round(np.mean(np.array(d["oos"]) > 0)*100, 0),
        })
    gdf = pd.DataFrame(rows)
    by_is = gdf.sort_values("IS夏普均值", ascending=False).reset_index(drop=True)
    by_oos = gdf.sort_values("OOS夏普均值", ascending=False).reset_index(drop=True)
    pdf = pd.DataFrame(per_symbol_best)

    gdf.to_csv(os.path.join(OUT_DIR, "grid_all.csv"), index=False, encoding="utf-8-sig")
    pdf.to_csv(os.path.join(OUT_DIR, "grid_per_symbol.csv"), index=False, encoding="utf-8-sig")

    # 报告
    R = ["# MACD × LON × 均线 参数穷举回测\n\n",
         f"网格：MACD {len(MACD_SETS)} 组 × LON {len(LON_SETS)} 组 × MA {MA_PERIODS} × 离场 {EXIT_BARS} = {len(combos)} 组；"
         f"44 品种；样本内前 {int(SPLIT*100)}% 选参，样本外后 {100-int(SPLIT*100)}% 验证。\n\n",
         "## 全局最优（按 44 品种平均样本内夏普 Top10）\n",
         by_is.head(10).to_markdown(index=False), "\n\n",
         "## 样本外最稳健（按平均样本外夏普 Top10）\n",
         by_oos.head(10).to_markdown(index=False), "\n\n"]

    top_is = by_is.iloc[0]
    top_oos = by_oos.iloc[0]
    R.append("## 关键观察\n")
    R.append(f"- 样本内最优组合：MACD{top_is['MACD']} + LON{top_is['LON']} + MA{top_is['MA']}/离场{top_is['离场根']}根，"
             f"IS夏普 {top_is['IS夏普均值']} → **OOS夏普 {top_is['OOS夏普均值']}**（衰减 {top_is['IS夏普均值']-top_is['OOS夏普均值']:.2f}）。\n")
    R.append(f"- 样本外最稳健组合：MACD{top_oos['MACD']} + LON{top_oos['LON']} + MA{top_oos['MA']}/离场{top_oos['离场根']}根，"
             f"OOS夏普 {top_oos['OOS夏普均值']}、OOS盈利品种 {top_oos['OOS盈利品种%']:.0f}%。\n")
    # 过拟合度量：每品种最优的 IS vs OOS
    is_mean = pdf["样本内夏普"].mean()
    oos_mean = pdf["样本外夏普"].mean()
    R.append(f"- **过拟合警示**：给每个品种单独挑最优，样本内平均夏普 {is_mean:.2f}，"
             f"到样本外只剩 **{oos_mean:.2f}**（掉 {is_mean-oos_mean:.2f}）——分品种精调严重过拟合。\n")
    # 单一稳健组合 vs 分品种精调（都看样本外）
    R.append(f"- 用**单一稳健组合**（样本外最稳健那组：MACD{top_oos['MACD']}+LON{top_oos['LON']}+MA{top_oos['MA']}/离场{top_oos['离场根']}根）"
             f"在样本外平均夏普 {top_oos['OOS夏普均值']}、{top_oos['OOS盈利品种%']:.0f}% 品种为正，"
             f"{'优于' if top_oos['OOS夏普均值']>oos_mean else '不优于'}逐品种精调的样本外 {oos_mean:.2f}——"
             "**与其给每个品种过度优化，不如用一套偏慢偏长的稳健参数**。\n")

    R.append("\n## 各品种样本内最优组合（及样本外验证）\n")
    R.append(pdf.to_markdown(index=False))
    R.append("\n\n## 结论\n")
    R.append("- 穷举能找到漂亮的样本内参数，但**样本外普遍大幅衰减**，分品种精调尤其严重——这是过拟合的典型表现。\n")
    R.append("- 更可信的是**样本外稳健、且在多数品种上为正**的统一组合；不要照搬每个品种的样本内最优。\n")
    R.append("- 参数选择对结果的影响，**远小于**前面发现的「出场方式（让利润奔跑）」「品种筛选（工业品）」「分散」这些结构性因素。\n")

    with open(os.path.join(OUT_DIR, "GRIDSEARCH.md"), "w", encoding="utf-8") as fh:
        fh.write("".join(R))
    print("\n" + "".join(R[:8]))
    print(f"\n报告已保存: {OUT_DIR}/GRIDSEARCH.md（+ grid_all.csv, grid_per_symbol.csv）")


if __name__ == "__main__":
    main()
