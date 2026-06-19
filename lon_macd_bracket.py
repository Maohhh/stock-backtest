#!/usr/bin/env python3
"""
固定「+TP 点止盈 / -SL 点止损」括号单的胜率与期望（点为绝对价格点，非百分比）

打法：按 v1 指标进场（次日开盘价 P），挂 止盈=P±TP、止损=P∓SL 的括号单，先碰到哪个就走。
保守假设：同一根 K 线若同时触及止盈与止损，按**先止损**处理（不高估）。
手续费按名义额双边折算成点数从期望里扣除。

输出每品种：止盈/止损/未触发 笔数、胜率、点数期望、% 期望，以及按复利的近似总收益。

用法：
    python lon_macd_bracket.py                 # 默认 TP=10 SL=5，玻璃+代表品种
    python lon_macd_bracket.py FG0 10 5         # 指定品种、止盈点、止损点
    python lon_macd_bracket.py FG0 10 5 RB0 CU0
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
    sig = generate_signals(df, ma_period=20, exit_bars=3)
    pos = sig["state"].shift(1).fillna(0).to_numpy()
    n = len(pos)
    segs, i = [], 0
    while i < n:
        if pos[i] == 0:
            i += 1
            continue
        s = pos[i]
        start = i
        while i < n and pos[i] == s:
            i += 1
        segs.append((int(s), start, i - 1))
    return segs


def bracket(df: pd.DataFrame, tp: float, sl: float, max_bars: int = 120) -> dict:
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    low = df["low"].to_numpy()
    c = df["close"].to_numpy()
    n = len(df)
    segs = segments(df)

    n_tp = n_sl = n_none = 0
    pnls = []          # 每笔盈亏（点，毛）
    avg_price = float(np.nanmean(c))
    for side, start, _end in segs:
        entry = o[start]
        tp_px = entry + tp * side
        sl_px = entry - sl * side
        outcome = None
        end = min(start + max_bars, n)
        for j in range(start, end):
            hit_sl = (low[j] <= sl_px) if side == 1 else (h[j] >= sl_px)
            hit_tp = (h[j] >= tp_px) if side == 1 else (low[j] <= tp_px)
            if hit_sl:           # 保守：同根先止损
                outcome = -sl
                n_sl += 1
                break
            if hit_tp:
                outcome = tp
                n_tp += 1
                break
        if outcome is None:
            outcome = (c[end - 1] - entry) * side
            n_none += 1
        pnls.append(outcome)

    pnls = np.array(pnls) if pnls else np.array([0.0])
    nseg = len(pnls)
    comm_pts = avg_price * COMMISSION * 2  # 双边手续费折点
    net = pnls - comm_pts
    win_rate = n_tp / nseg if nseg else 0.0
    return {
        "n": nseg, "n_tp": n_tp, "n_sl": n_sl, "n_none": n_none,
        "win_rate": win_rate,
        "exp_gross_pts": float(pnls.mean()),
        "exp_net_pts": float(net.mean()),
        "exp_net_pct": float(net.mean()) / avg_price,
        "comm_pts": comm_pts,
        "avg_price": avg_price,
        "tp_pct": tp / avg_price, "sl_pct": sl / avg_price,
        "total_net_pct_compounded": float(np.prod(1 + net / avg_price) - 1),
    }


def main(argv):
    ensure_data()
    args = argv[:]
    nums = [a for a in args if a.replace(".", "").isdigit()]
    syms = [a for a in args if not a.replace(".", "").isdigit()]
    tp = float(nums[0]) if len(nums) >= 1 else 10.0
    sl = float(nums[1]) if len(nums) >= 2 else 5.0
    if not syms:
        syms = ["FG0", "RB0", "CU0", "I0", "M0", "IF0", "AG0", "TA0"]
    avail = set(f[:-4] for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
    syms = [s for s in syms if s in avail]

    R = ["# 固定止盈止损括号单：+{:g} 点止盈 / -{:g} 点止损\n".format(tp, sl),
         f"盈亏比 = {tp/sl:.1f} : 1；盈亏平衡胜率 = {sl/(tp+sl)*100:.1f}%。\n",
         "保守假设：同根 K 线同时触及止盈止损按**先止损**算；已扣双边手续费。\n\n",
         "| 品种 | 均价 | 止盈%/止损% | 笔数 | 止盈/止损/未触发 | **胜率** | 毛期望(点) | 净期望(点) | 净期望% |\n",
         "|---|---|---|---|---|---|---|---|---|\n"]
    detail = {}
    for s in syms:
        r = bracket(load_symbol(s), tp, sl)
        detail[s] = r
        R.append(
            f"| {s} | {r['avg_price']:.0f} | {r['tp_pct']*100:.2f}%/{r['sl_pct']*100:.2f}% | {r['n']} | "
            f"{r['n_tp']}/{r['n_sl']}/{r['n_none']} | **{r['win_rate']*100:.1f}%** | "
            f"{r['exp_gross_pts']:+.1f} | {r['exp_net_pts']:+.1f} | {r['exp_net_pct']*100:+.3f}% |\n"
        )
    R.append("\n")

    if "FG0" in detail:
        r = detail["FG0"]
        be = sl / (tp + sl)
        R.append("## 结论（以玻璃 FG0 为例）\n")
        R.append(
            f"- 玻璃 +{tp:g}/-{sl:g} 点：胜率 **{r['win_rate']*100:.1f}%**"
            f"（止盈 {r['n_tp']} 笔 / 止损 {r['n_sl']} 笔 / 未触发 {r['n_none']} 笔）。\n"
        )
        be_ok = r["win_rate"] > be
        R.append(
            f"- 盈亏比 2:1，盈亏平衡只需胜率 {be*100:.1f}%；实际胜率 {r['win_rate']*100:.1f}% "
            f"{'高于' if be_ok else '低于'}盈亏平衡，**净期望 {r['exp_net_pts']:+.1f} 点/笔（{r['exp_net_pct']*100:+.3f}%）**。\n"
        )
        if be_ok:
            R.append("- 这次是**正期望**：因为 2:1 的盈亏比把盈亏平衡线压到 33%，进场只要不是太差就能过线。"
                     "比「赚3点就走」那种 1:35 的烂盈亏比健康得多——**止损比止盈紧才是关键**。\n")
        else:
            R.append("- 仍是负期望：进场短期偏随机 + 5 点止损过近，容易被噪音扫损。\n")
        R.append("- 注意点数是绝对值：对铜/股指等高价品种，10/5 点≈0.01%，基本是手续费和噪音级别，参考意义小。\n")

    out = os.path.join(OUT_DIR, "BRACKET.md")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("".join(R))
    print("".join(R))
    print(f"报告已保存: {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
