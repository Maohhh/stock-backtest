"""
期货 15 分钟横截面反转指标 —— 端到端运行脚本

流程:
1. 若本地无缓存，则从新浪下载能化板块 15 分钟数据
2. 计算板块内横截面反转目标权重
3. 回测：毛收益 / 扣成本净值 / 波动率目标加杠杆
4. 打印绩效、做窗口内稳健性切分、保存净值曲线图

用法:
    python run_futures_indicator.py [板块名]   # 默认 energy_chem
"""

import os
import sys

import numpy as np
import pandas as pd

from src.data.futures import SECTORS, download_sector, load_panel
from src.strategies.futures_xs_reversal import FuturesXSReversalStrategy, optimize
from src.backtest.futures_engine import run_panel_backtest, bars_per_year

DATA_DIR = "data/futures_15min"
COST = 0.0002          # 单边 2bp（手续费+滑点）
FORM = 4               # 形成期：4 根 = 1 小时
REBALANCE = 24         # 再平衡间隔：24 根 ≈ 1.5 个交易日
TARGET_VOL = 0.25      # 波动率目标 25%（期货天然杠杆，约 2~3x）


def ensure_data(symbols, sector):
    have = [s for s in symbols if os.path.exists(os.path.join(DATA_DIR, f"{s}.csv"))]
    if len(have) < max(3, len(symbols) // 2):
        print(f"本地数据不足，正在下载板块 {sector} 的 15 分钟数据 ...")
        download_sector(sector, freq="15min", out_dir=DATA_DIR)


def half_split_stability(prices, strat):
    """窗口内前后两半各自回测，检验稳健性（非严格样本外，但能看符号是否稳定）。"""
    cut = len(prices) // 2
    a = strat.run(prices.iloc[:cut])
    b = strat.run(prices.iloc[cut:])
    return a, b


def main():
    sector = sys.argv[1] if len(sys.argv) > 1 else "ferrous"
    symbols = SECTORS[sector]
    ensure_data(symbols, sector)
    prices = load_panel(symbols, data_dir=DATA_DIR)
    bpy = bars_per_year(prices)

    print("=" * 68)
    print(f"板块: {sector}  品种: {list(prices.columns)}")
    print(f"K线: {len(prices)} 根 | 交易日: {len(pd.Index(prices.index).normalize().unique())}"
          f" | 区间: {prices.index[0]} ~ {prices.index[-1]}")
    print(f"年化用 bar/年 ≈ {bpy:.0f}")
    print("=" * 68)

    # 1) 毛收益（每根再平衡，展示原始 alpha 强度）
    gross = FuturesXSReversalStrategy(form=1, rebalance=1, cost=0.0).run(prices)
    print("\n[原始 alpha] form=1 每根再平衡 无成本（仅展示信号强度，不可交易）")
    print("  ", gross.summary())

    # 2) 净值：低换手 + 成本（可交易主结果）
    strat = FuturesXSReversalStrategy(form=FORM, rebalance=REBALANCE, cost=COST)
    net = strat.run(prices)
    print(f"\n[可交易净值] form={FORM} rebalance={REBALANCE} 成本={COST*1e4:.0f}bp")
    print("  ", net.summary())

    # 3) 波动率目标加杠杆（高年化版本）
    levered = FuturesXSReversalStrategy(form=FORM, rebalance=REBALANCE,
                                        cost=COST, target_vol=TARGET_VOL).run(prices)
    print(f"\n[高年化] 在净值基础上做 {TARGET_VOL*100:.0f}% 波动率目标（期货杠杆）")
    print("  ", levered.summary())

    # 4) 窗口内稳健性
    a, b = half_split_stability(prices, strat)
    print("\n[稳健性] 窗口内前后两半（同参数，扣成本）")
    print(f"   前半: 夏普={a.sharpe:.2f} 年化={a.ann_return*100:.0f}%")
    print(f"   后半: 夏普={b.sharpe:.2f} 年化={b.ann_return*100:.0f}%")

    # 5) 参数网格（研究参考）
    print("\n[参数网格] 按夏普排序前 6")
    print(optimize(prices, cost=COST).head(6).to_string(index=False))

    # 6) 画图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(11, 8))
        gross.equity.plot(ax=ax[0], color="crimson",
                          label=f"Raw alpha (no cost)  ann={gross.ann_return*100:.0f}% SR={gross.sharpe:.1f}")
        net.equity.plot(ax=ax[0], color="steelblue",
                        label=f"Tradable net 2bp  ann={net.ann_return*100:.0f}% SR={net.sharpe:.1f}")
        ax[0].set_title(f"{sector} 15min Cross-Sectional Reversal - Equity")
        ax[0].legend(); ax[0].grid(alpha=.3)
        levered.equity.plot(ax=ax[1], color="green",
                            label=f"Vol-target {TARGET_VOL*100:.0f}% (lev {levered.leverage:.1f}x)  "
                                  f"ann={levered.ann_return*100:.0f}% maxDD={levered.max_drawdown*100:.0f}%")
        ax[1].set_title("High-annualized version (volatility targeting)")
        ax[1].legend(); ax[1].grid(alpha=.3)
        plt.tight_layout()
        out = f"data/{sector}_xs_reversal.png"
        plt.savefig(out, dpi=110)
        print(f"\n净值曲线已保存: {out}")
    except Exception as e:
        print(f"\n（绘图跳过: {e}）")


if __name__ == "__main__":
    main()
