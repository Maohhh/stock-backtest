#!/usr/bin/env python3
"""
固定「+N 个点」止盈的胜率研究（N 为绝对价格点，不是百分比）

打法：按 v1 指标进场（次日开盘价 P 进场），只要价格在持仓方向上摸到 P±N 点就止盈离场；
若在策略「正常离场」（连续 3 根破 20 日均线）之前没摸到，就认为这笔没达成止盈，
按正常离场价了结。统计：
    - 命中率 = 在正常离场前摸到 +N 点的比例（即「赚 N 点就走」的胜率）；
    - 未命中单子的了结盈亏（点）与持仓中最大不利 MAE（点），揭示「高胜率的代价」。

注意：N 是绝对价格点，对不同品种含义不同（脚本会换算成 % 参考）。

用法：
    python lon_macd_tp_points.py            # 玻璃 + 一组代表品种
    python lon_macd_tp_points.py FG0 3       # 指定品种与点数
    python lon_macd_tp_points.py FG0 1,3,5,10  # 指定多个点数
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lon_macd_backtest import DATA_DIR, ensure_data, load_symbol, COMMISSION  # noqa: E402
from src.strategies.lon_macd_strategy import generate_signals  # noqa: E402

OUT_DIR = "lon_macd_results"


def segments(df: pd.DataFrame):
    """用 v1（MA20）生成持仓段：返回 (side, start_bar, end_bar) 列表。start 为首根持仓 K 线。"""
    sig = generate_signals(df, ma_period=20, exit_bars=3)
    pos = sig["state"].shift(1).fillna(0).to_numpy()
    n = len(pos)
    segs = []
    i = 0
    while i < n:
        if pos[i] == 0:
            i += 1
            continue
        s = pos[i]
        start = i
        while i < n and pos[i] == s:
            i += 1
        segs.append((int(s), start, i - 1))
    return sig, segs


def study(df: pd.DataFrame, points: float) -> dict:
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    low = df["low"].to_numpy()
    c = df["close"].to_numpy()
    _, segs = segments(df)

    hit = 0
    miss_pnls = []     # 未命中单子的了结盈亏（点）
    maes = []          # 每段持仓中最大不利（点）
    bars_to_hit = []
    for side, start, end in segs:
        entry = o[start]
        target = entry + points * side
        seg_hit = False
        for j in range(start, end + 1):
            # 段内最大不利（点）
            adverse = (low[j] - entry) if side == 1 else (entry - h[j])
            maes.append(adverse)
            fav_extreme = h[j] if side == 1 else low[j]
            reached = (fav_extreme >= target) if side == 1 else (fav_extreme <= target)
            if reached:
                seg_hit = True
                bars_to_hit.append(j - start + 1)
                break
        if seg_hit:
            hit += 1
        else:
            miss_pnls.append((c[end] - entry) * side)

    nseg = len(segs)
    avg_price = float(np.nanmean(c))
    miss = np.array(miss_pnls) if miss_pnls else np.array([0.0])
    return {
        "n": nseg,
        "hit_rate": hit / nseg if nseg else 0.0,
        "points": points,
        "pct_equiv": points / avg_price,
        "avg_bars_to_hit": float(np.mean(bars_to_hit)) if bars_to_hit else float("nan"),
        "miss_count": len(miss_pnls),
        "miss_avg_pnl_pts": float(miss.mean()),
        "miss_worst_pts": float(miss.min()),
        "expectancy_pts": (hit * points + miss.sum()) / nseg if nseg else 0.0,
        "avg_price": avg_price,
    }


def main(argv):
    ensure_data()
    args = [a for a in argv if not a.startswith("--")]
    # 解析品种与点数
    syms, pts = [], [1, 3, 5, 10]
    for a in args:
        if a.replace(",", "").replace(".", "").isdigit():
            pts = [float(x) for x in a.split(",")]
        else:
            syms.append(a)
    if not syms:
        syms = ["FG0", "RB0", "CU0", "I0", "M0", "IF0", "AG0", "TA0"]

    avail = set(f[:-4] for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
    syms = [s for s in syms if s in avail]

    report = ["# 固定「+N 个点」止盈的胜率\n",
              "打法：v1 指标进场（次日开盘），先摸到 +N 个**绝对价格点**就止盈；正常离场前没摸到就按离场价了结。\n",
              "（N 是价格点，不是百分比；同样 3 点对不同品种 % 含义不同，已换算参考。）\n\n"]

    for sym in syms:
        df = load_symbol(sym)
        report.append(f"## {sym}（均价≈{study(df, pts[0])['avg_price']:.0f}）\n")
        report.append("| +N点 | ≈% | 进场样本 | **赚N点胜率** | 平均几根摸到 | 未命中均盈亏(点) | 未命中最差(点) | 期望(点/笔) |\n")
        report.append("|---|---|---|---|---|---|---|---|\n")
        for p in pts:
            r = study(df, p)
            report.append(
                f"| {p:g} | {r['pct_equiv']*100:.2f}% | {r['n']} | **{r['hit_rate']*100:.1f}%** | "
                f"{r['avg_bars_to_hit']:.1f} | {r['miss_avg_pnl_pts']:+.0f} | {r['miss_worst_pts']:+.0f} | "
                f"{r['expectancy_pts']:+.1f} |\n"
            )
        report.append("\n")

    # 结论（以玻璃 3 点为例）
    if "FG0" in syms:
        r3 = study(load_symbol("FG0"), 3)
        report.append("## 结论\n")
        report.append(
            f"- 以玻璃 +3 点为例：胜率高达 **{r3['hit_rate']*100:.0f}%**，平均 {r3['avg_bars_to_hit']:.1f} 根 K 线就摸到——"
            "「赚 3 个点就走」确实**绝大多数都能达成**。\n"
        )
        report.append(
            f"- **但这是个陷阱**：剩下没摸到的 {r3['miss_count']} 笔，平均亏 {abs(r3['miss_avg_pnl_pts']):.0f} 点、"
            f"最惨一笔亏 {abs(r3['miss_worst_pts']):.0f} 点；每笔期望只有 **{r3['expectancy_pts']:+.1f} 点**。"
            "用 3 点小利去扛上百点的反向风险，是典型「高胜率、负期望/易爆仓」的卖法。\n"
        )
        report.append(
            "- 点数越小胜率越高（噪音都能摸到）、越大胜率越低；但小止盈把右尾大赢全砍掉，"
            "和这套趋势指标「靠大波段赚钱」的逻辑正好相反。\n"
        )

    out = os.path.join(OUT_DIR, "TP_POINTS.md")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("".join(report))
    print("".join(report))
    print(f"报告已保存: {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
