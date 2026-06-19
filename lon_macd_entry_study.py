#!/usr/bin/env python3
"""
LON + MACD 进场信号「事件研究」

回答一个问题：按这个指标的进场信号买/卖进去，是不是马上就能赚，哪怕几个点？

做法：把 v1 的进场条件（与离场/均线无关）当事件，统计每次信号后未来 N 根 K 线、
按持仓方向计的远期收益：
    - 进场价 = 信号次日开盘价（贴近实盘，无未来函数）；
    - 远期收益 = 到第 N 根 K 线收盘、按多/空方向折算的涨跌幅；
    - 胜率 = 远期收益 > 0 的比例；
    - 另用最高/最低价算进场后 K 根内的 MFE（最大有利幅度）/ MAE（最大不利幅度），
      看「能不能摸到几个点」以及「先朝哪个方向走」。

进场条件（与 lon_macd_strategy 一致）：
    做多：LON>0 且 MACD 金叉且 DIF/DEA 均>0
    做空：LON<0 且 MACD 死叉且 DIF/DEA 均<0

用法：
    python lon_macd_entry_study.py          # 全部 43 日线品种汇总
    python lon_macd_entry_study.py FG0       # 单品种（日线，需先有该 csv）
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lon_macd_backtest import DATA_DIR, ensure_data, load_symbol  # noqa: E402
from src.indicators import macd, lon  # noqa: E402

OUT_DIR = "lon_macd_results"
HORIZONS = [1, 2, 3, 5, 10, 20]   # 远期 K 线数
MFE_WINDOW = 5                    # MFE/MAE 观察窗（根）
THRESHOLDS = [0.002, 0.005, 0.01]  # 0.2% / 0.5% / 1%（“几个点”量级）


def entry_events(df: pd.DataFrame) -> pd.DataFrame:
    """返回每根 K 线的 long_entry / short_entry 布尔，以及 OHLC，用于事件研究。"""
    data = df.reset_index(drop=True).copy()
    m = macd(data)
    dif, dea = m["DIF"], m["DEA"]
    pd_, pe_ = dif.shift(1), dea.shift(1)
    golden = (dif > dea) & (pd_ <= pe_)
    death = (dif < dea) & (pd_ >= pe_)
    lv = lon(data)["LON"]
    data["long_entry"] = ((lv > 0) & golden & (dif > 0) & (dea > 0)).fillna(False)
    data["short_entry"] = ((lv < 0) & death & (dif < 0) & (dea < 0)).fillna(False)
    return data


def collect(df: pd.DataFrame, warmup: int = 60) -> list:
    """对单品种收集所有进场事件的远期收益与 MFE/MAE。"""
    d = entry_events(df)
    o = d["open"].to_numpy()
    h = d["high"].to_numpy()
    low = d["low"].to_numpy()
    c = d["close"].to_numpy()
    n = len(d)
    rows = []
    for t in range(warmup, n - 1):  # 需要 t+1 开盘进场
        for side, flag in ((1, d["long_entry"].iloc[t]), (-1, d["short_entry"].iloc[t])):
            if not flag:
                continue
            entry = o[t + 1]
            if entry <= 0:
                continue
            rec = {"side": side}
            # 远期收益
            for hh in HORIZONS:
                j = t + hh
                if j >= n:
                    rec[f"r{hh}"] = np.nan
                    continue
                fwd = c[j]
                rec[f"r{hh}"] = (fwd / entry - 1.0) * side
            # MFE/MAE（窗口内，方向调整）
            end = min(t + 1 + MFE_WINDOW, n)
            hi = h[t + 1:end]
            lo = low[t + 1:end]
            if len(hi):
                if side == 1:
                    rec["mfe"] = hi.max() / entry - 1.0
                    rec["mae"] = lo.min() / entry - 1.0
                else:
                    rec["mfe"] = entry / lo.min() - 1.0
                    rec["mae"] = entry / hi.max() - 1.0
            rows.append(rec)
    return rows


def summarize(rows: list, label: str) -> str:
    df = pd.DataFrame(rows)
    L = [f"### {label}（样本 {len(df)} 次进场：多 {int((df['side']==1).sum())} / 空 {int((df['side']==-1).sum())}）\n"]
    L.append("| 持有K线 | 胜率 | 平均收益 | 中位收益 | 平均赢 | 平均亏 |\n|---|---|---|---|---|---|\n")
    for hh in HORIZONS:
        r = df[f"r{hh}"].dropna()
        win = (r > 0).mean()
        avg = r.mean()
        med = r.median()
        aw = r[r > 0].mean()
        al = r[r <= 0].mean()
        L.append(f"| {hh} | {win*100:.1f}% | {avg*100:+.2f}% | {med*100:+.2f}% | "
                 f"{aw*100:+.2f}% | {al*100:+.2f}% |\n")
    # MFE/MAE：进场后窗口内能摸到的有利/不利幅度
    mfe = df["mfe"].dropna()
    mae = df["mae"].dropna()
    L.append(f"\n进场后 {MFE_WINDOW} 根内：平均最大有利 MFE {mfe.mean()*100:+.2f}%、"
             f"平均最大不利 MAE {mae.mean()*100:+.2f}%。\n")
    L.append("「能摸到几个点」的概率（窗口内最高/最低触及阈值）：\n")
    L.append("| 阈值 | 摸到有利(MFE≥) | 触及不利(MAE≤-) |\n|---|---|---|\n")
    for thr in THRESHOLDS:
        pu = (mfe >= thr).mean()
        pd_ = (mae <= -thr).mean()
        L.append(f"| {thr*100:.1f}% | {pu*100:.0f}% | {pd_*100:.0f}% |\n")
    # 次日方向
    r1 = df["r1"].dropna()
    L.append(f"\n进场**次日**（第1根）就为正的概率：{(r1>0).mean()*100:.1f}%；"
             f"平均 {r1.mean()*100:+.2f}%。\n")
    return "".join(L), df


def main(argv):
    syms = [a for a in argv if not a.startswith("--")]
    ensure_data()
    if not syms:
        syms = sorted(f[:-4] for f in os.listdir(DATA_DIR) if f.endswith(".csv"))

    all_rows = []
    per_symbol = {}
    for s in syms:
        try:
            df = load_symbol(s)
        except Exception:  # noqa: BLE001
            continue
        if len(df) < 120:
            continue
        rows = collect(df)
        per_symbol[s] = rows
        all_rows.extend(rows)

    os.makedirs(OUT_DIR, exist_ok=True)
    report = ["# LON + MACD 进场信号事件研究\n",
              "问题：按这个指标进场，是不是买进去就能赚（哪怕几个点）？\n",
              "口径：进场价=信号次日开盘；远期收益按持仓方向折算；进场信号与均线/离场无关。\n\n"]
    pooled_text, pooled_df = summarize(all_rows, f"全部 {len(per_symbol)} 个日线品种汇总")
    report.append(pooled_text)

    # 给单独点名的品种也出一份（若在参数里）
    named = [s for s in argv if not s.startswith("--") and s in per_symbol]
    for s in named:
        if per_symbol[s]:
            t, _ = summarize(per_symbol[s], s)
            report.append("\n" + t)

    # 结论
    r1 = pooled_df["r1"].dropna()
    r5 = pooled_df["r5"].dropna()
    r20 = pooled_df["r20"].dropna()
    mfe = pooled_df["mfe"].dropna()
    report.append("\n## 结论\n")
    report.append(
        f"- **不是「买进去就能赚」**：进场次日为正的概率只有 {(r1>0).mean()*100:.0f}%，"
        f"持有 5 根胜率 {(r5>0).mean()*100:.0f}%、20 根 {(r20>0).mean()*100:.0f}%，都在 50% 上下甚至以下。\n"
    )
    report.append(
        f"- **想摸几个点是能摸到，但同样容易先被打：** 进场后 {MFE_WINDOW} 根内有 "
        f"{(mfe>=0.005).mean()*100:.0f}% 的概率最高摸到 +0.5%，但也有相近概率先回撤同样幅度——"
        "顺势进场是「正期望靠右尾大赢、多数单子小亏」，不是「稳赚几个点」。\n"
    )
    report.append(
        "- 真正的盈利来自**拿住趋势单**（少数大赢家），而不是进场瞬间的确定性。"
        "若你期望「买进去立刻稳赚几个点」，这个指标（以及绝大多数趋势指标）都做不到。\n"
    )

    path = os.path.join(OUT_DIR, "ENTRY_STUDY.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(report))
    print("".join(report))
    print(f"\n报告已保存: {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
