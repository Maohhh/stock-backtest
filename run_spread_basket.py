"""
跨品种套利篮子 —— 端到端运行（穷尽对比跨品种 vs 跨期后的最优方案）

1. 下载所需单合约（具体交割月，含已退市，回溯约 2018）
2. 重构同月跨品种价差（=券商挂牌套利指令真身）
3. 各自 Z 回归 + 风险平价组合，输出胜率/盈亏比/盈利因子/合并夏普 + 净值图

用法: python run_spread_basket.py
"""

import os

import numpy as np
import pandas as pd

from src.data.contracts import download_product_contracts, matched_spread_segments, calendar_spread_segments
from src.strategies.spread_basket import SpreadBasket, backtest_segments

# 篮子成分：(名称, 品种A, 品种B, 价差模式) —— 选低相关的协整对
PAIRS = [
    ("玉米-淀粉", "C", "CS", "diff"),
    ("螺纹-热卷", "RB", "HC", "diff"),
    ("铁矿-螺纹", "I", "RB", "ratio"),
    ("豆粕-菜粕", "M", "RM", "ratio"),
]
PRODUCTS = sorted({p for _, a, b, _ in PAIRS for p in (a, b)})


def ensure_data():
    have = len([f for f in os.listdir("data/futures_contracts")]) if os.path.isdir("data/futures_contracts") else 0
    if have < 80:
        print("下载单合约数据（首次较慢）...")
        for p in PRODUCTS:
            n = download_product_contracts(p)
            print(f"  {p}: {len(n)} 个合约")


def main():
    ensure_data()
    legs = {nm: matched_spread_segments(a, b, mode=md) for nm, a, b, md in PAIRS}

    print("=" * 60)
    print("各跨品种套利对 Z 回归（同月价差，成本 4 点/次）")
    print("=" * 60)
    print(f"{'套利对':<12}{'笔数':>5}{'胜率':>6}{'盈亏比':>7}{'盈利因子':>8}")
    for nm in legs:
        _, st = backtest_segments(legs[nm])
        print(f"{nm:<12}{st['n_trades']:>5}{st['win_rate']*100:>5.0f}%"
              f"{st['payoff']:>7.2f}{st['profit_factor']:>8.2f}")

    basket = SpreadBasket(legs).run()
    print("\n各对月度PnL相关性（低/负 = 可分散）:")
    print(basket["corr"].round(2).to_string())
    print("\n各对 年化夏普:")
    for c, s in basket["leg_ann_sharpe"].items():
        print(f"  {c:<12} {s:.2f}")
    print(f"\n>>> 风险平价组合: 年化夏普={basket['combo_ann_sharpe']:.2f} "
          f"盈利月占比={basket['combo_win_month']*100:.0f}% "
          f"月数={len(basket['combo_monthly'])}")

    # 对比：跨期套利（同方法，仅供对照——已知整体弱于跨品种）
    print("\n[对照] 跨期套利（相邻月近-远价差 Z 回归）:")
    for p in ["C", "RB", "M", "I"]:
        _, st = backtest_segments(calendar_spread_segments(p))
        if st["n_trades"] >= 5:
            print(f"  {p:<4} 胜率={st['win_rate']*100:>3.0f}% 盈利因子={st['profit_factor']:.2f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 6))
        m = basket["monthly"]
        ascii_label = {nm: f"{a}-{b}" for nm, a, b, _ in PAIRS}
        for c in m.columns:
            (m[c] / m[c].std()).cumsum().plot(ax=ax, alpha=0.5, label=ascii_label.get(c, c))
        basket["combo_equity"].plot(ax=ax, color="black", lw=2.5,
                                    label=f"Basket (SR={basket['combo_ann_sharpe']:.2f}, "
                                          f"win-month={basket['combo_win_month']*100:.0f}%)")
        ax.set_title("Cross-product spread arbitrage basket (risk-parity, monthly equity)")
        ax.legend(); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig("data/spread_basket.png", dpi=110)
        print("\n净值图已保存: data/spread_basket.png")
    except Exception as e:
        print(f"（绘图跳过: {e}）")


if __name__ == "__main__":
    main()
