"""从真实数据里挑一笔干净的 ORB 做多例子, 画成带标注的示意图。"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

matplotlib.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from src.indicators.opening_range_breakout import opening_range_breakout

PROD = sys.argv[1] if len(sys.argv) > 1 else "RB"
OR_BARS = 2

df = pd.read_parquet(f"data_futures/weighted_15min/{PROD}.parquet").reset_index(drop=True)
sig = opening_range_breakout(df, or_bars=OR_BARS)

# 找一个: 有做多信号、所在会话不太长、且进场后涨了一段(讲解用赢家)
best = None
for sess_id, idx in sig.groupby("session").groups.items():
    idx = list(idx)
    if len(idx) < 12 or len(idx) > 26:
        continue
    sub = sig.loc[idx]
    longs = sub.index[sub["long_signal"]].tolist()
    if not longs:
        continue
    si = longs[0]
    pos = idx.index(si)
    if pos + 1 >= len(idx):
        continue
    entry = df["open"].iloc[idx[pos + 1]]
    stop = sig["stop_long"].iloc[si]
    risk = entry - stop
    if risk <= 0:
        continue
    after = df.iloc[idx[pos + 1]:idx[-1] + 1]
    run = after["high"].max() - entry
    min_low = after["low"].min()
    # 干净的"突破即走": 跑出>1.8R, 且进场后没回踩到接近止损(留 0.5R 余量)
    if run / risk > 1.8 and min_low > stop + 0.5 * risk:
        best = (idx, si, pos, entry, stop)
        break

if best is None:
    print("未找到合适例子"); sys.exit(1)

idx, si, pos, entry, stop = best
sub = df.loc[idx].reset_index(drop=True)
ssig = sig.loc[idx].reset_index(drop=True)
x = np.arange(len(sub))

or_high = ssig["or_high"].iloc[0]
or_low = ssig["or_low"].iloc[0]
entry_x = pos + 1
exit_x = len(sub) - 1
exit_px = sub["close"].iloc[-1]

fig, ax = plt.subplots(figsize=(13, 7))
# K线
for i in range(len(sub)):
    o, h, l, c = sub.loc[i, ["open", "high", "low", "close"]]
    color = "#d33" if c >= o else "#3a3"
    ax.plot([i, i], [l, h], color=color, linewidth=1, zorder=2)
    ax.add_patch(Rectangle((i - 0.3, min(o, c)), 0.6, abs(c - o) + 1e-6,
                           facecolor=color, edgecolor=color, zorder=3))
# VWAP
ax.plot(x, ssig["vwap"], color="#06c", lw=1.4, ls="-", label="VWAP(当日均价)", zorder=4)
# 开盘区间盒子
ax.add_patch(Rectangle((-0.5, or_low), OR_BARS, or_high - or_low,
                       facecolor="#ffd", edgecolor="#cc0", lw=1.5, zorder=1))
ax.axhline(or_high, color="#cc0", lw=1.2, ls="--", zorder=1)
ax.axhline(or_low, color="#cc0", lw=1.2, ls="--", zorder=1)
ax.text(-0.4, or_high, " 开盘区间上沿(突破=做多)", va="bottom", color="#990", fontsize=10)
ax.text(-0.4, or_low, " 开盘区间下沿", va="top", color="#990", fontsize=10)

# 进场 / 止损 / 出场
ax.scatter([entry_x], [entry], marker="^", s=260, color="#d00", zorder=6)
ax.annotate("买入(放量突破上沿\n且站上VWAP)", (entry_x, entry),
            xytext=(entry_x + 0.6, entry - (or_high - or_low) * 2.2),
            fontsize=10, color="#d00",
            arrowprops=dict(arrowstyle="->", color="#d00"))
ax.axhline(stop, color="#888", lw=1.2, ls=":", zorder=1)
ax.text(len(sub) - 1, stop, "止损=区间下沿 ", va="bottom", ha="right", color="#555", fontsize=10)
ax.scatter([exit_x], [exit_px], marker="v", s=220, color="#06c", zorder=6)
ax.annotate("时段收盘了结\n(让利润奔跑)", (exit_x, exit_px),
            xytext=(exit_x - 3.5, exit_px + (or_high - or_low) * 0.6),
            fontsize=10, color="#06c",
            arrowprops=dict(arrowstyle="->", color="#06c"))

risk = entry - stop
reward = exit_px - entry
ax.set_title(f"开盘区间突破(ORB)示意 — {PROD}  "
             f"风险 {risk:.0f} 点 / 盈利 {reward:.0f} 点 ≈ {reward/risk:.1f}R",
             fontsize=13)
ax.set_xlabel("一个交易时段内的 15 分钟 K 线"); ax.set_ylabel("价格")
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.25)
plt.tight_layout()
out = "data_futures/orb_example.png"
plt.savefig(out, dpi=110)
print("saved", out, "| 风险", round(risk), "盈利", round(reward), "R=", round(reward/risk, 2))
