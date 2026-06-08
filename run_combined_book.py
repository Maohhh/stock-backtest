"""
多板块横截面组合（高年化·低回撤）—— 端到端运行

把多个板块各自的"板块中性横截面反转"账本（彼此低相关）按等风险叠加，
再做波动率目标加杠杆。这是本仓库里**唯一在样本外/广度上稳健为正**的高年化方案
（单标的方向性指标已验证不成立，见 FUTURES_RESULTS.md 第 5 节）。

用法:
    python run_combined_book.py
"""

import numpy as np
import pandas as pd

from src.data.futures import SECTORS, load_panel
from src.indicators.xs_reversal import target_weights
from src.backtest.futures_engine import bars_per_year

# 你的真实成本：往返手续费 3 元(玉米基准≈单边 0.5bp) + 半跳滑点 ≈ 单边 2.24bp
COST = 0.000224
SECTORS_USED = ["ferrous", "energy_chem"]   # 低相关、横截面反转都稳健的两个板块
FORM = 4
REBALANCE = 24
TARGET_VOL = 0.25          # 默认保守档；上不封顶但实盘别超过此档太多（见文末风险提示）
DATA_DIR = "data/futures_15min"


def sector_book(sector: str) -> pd.Series:
    """单板块中性横截面反转账本的逐 bar 净收益（扣成本，未加杠杆）。"""
    px = load_panel(SECTORS[sector], data_dir=DATA_DIR)
    w = target_weights(px, form=FORM, rebalance=REBALANCE)
    ret = px.pct_change()
    pos = w.shift(1)
    return (pos * ret).sum(axis=1) - pos.diff().abs().sum(axis=1) * COST, bars_per_year(px)


def stats(pnl, bpy, lev=1.0):
    p = (pnl * lev).dropna()
    eq = (1 + p).cumprod()
    return {
        "年化": round(p.mean() * bpy * 100, 1),
        "夏普": round(p.mean() / p.std() * np.sqrt(bpy), 2),
        "最大回撤": round((eq / eq.cummax() - 1).min() * 100, 1),
        "年化波动": round(p.std() * np.sqrt(bpy) * 100, 1),
    }


def main():
    books, bpy = {}, None
    for sec in SECTORS_USED:
        pnl, bpy = sector_book(sec)
        books[sec] = pnl
        print(f"[{sec:11s}] {stats(pnl, bpy)}")

    P = pd.concat(books, axis=1).dropna()
    corr = P.corr()
    # 等风险（逆波动）叠加
    rw = (1 / P.std()); rw /= rw.sum()
    combo = (P * rw).sum(axis=1)
    print(f"\n板块间相关性:\n{corr.round(2).to_string()}")
    print(f"\n[组合·等风险] {stats(combo, bpy)}")

    vol0 = combo.std() * np.sqrt(bpy)
    print(f"\n[组合 + 波动率目标] 成本={COST*1e4:.2f}bp")
    tiers = [0.15, 0.25, 0.40, 0.60]
    for tgt in tiers:
        lev = tgt / vol0
        s = stats(combo, bpy, lev)
        flag = "  <= 默认/推荐" if abs(tgt - TARGET_VOL) < 1e-9 else ""
        print(f"  目标{tgt*100:>3.0f}%vol 杠杆{lev:>4.1f}x -> 年化{s['年化']:>6.1f}% "
              f"夏普{s['夏普']:.2f} 回撤{s['最大回撤']:>5.1f}%{flag}")

    # 图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        lev = TARGET_VOL / vol0
        fig, ax = plt.subplots(2, 1, figsize=(11, 8))
        for sec in SECTORS_USED:
            (1 + books[sec]).cumprod().plot(ax=ax[0], label=f"{sec} book")
        (1 + combo).cumprod().plot(ax=ax[0], color="black", lw=2,
                                   label=f"Combined (SR={stats(combo,bpy)['夏普']})")
        ax[0].set_title("Multi-sector cross-sectional books (unlevered, net)")
        ax[0].legend(); ax[0].grid(alpha=.3)
        s = stats(combo, bpy, lev)
        (1 + combo * lev).cumprod().plot(ax=ax[1], color="green",
            label=f"Combined vol-target {TARGET_VOL*100:.0f}% (lev {lev:.1f}x)  "
                  f"ann={s['年化']}% maxDD={s['最大回撤']}%")
        ax[1].set_title("High-annualized / low-drawdown deliverable")
        ax[1].legend(); ax[1].grid(alpha=.3)
        plt.tight_layout()
        out = "data/combined_book.png"
        plt.savefig(out, dpi=110)
        print(f"\n净值图已保存: {out}")
    except Exception as e:
        print(f"（绘图跳过: {e}）")


if __name__ == "__main__":
    main()
