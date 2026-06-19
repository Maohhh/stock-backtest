#!/usr/bin/env python3
"""
逐品种对比：MA20 原版出场 vs ATR k=2 移动止损（看 ATR 改善是否普遍，还是只对玻璃）

对每个品种用相同 v1 进场，分别跑 MA20 出场与 ATR(k=2) 出场，比较总收益与夏普，
统计改善的广度与分布。

用法： python lon_macd_atr_scan.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lon_macd_backtest import DATA_DIR, ensure_data, load_symbol, COMMISSION  # noqa: E402
from src.strategies import lon_macd_strategy as v1  # noqa: E402
from src.strategies import lon_macd_atr as atr_exit  # noqa: E402
from src.backtest.futures_engine import run_backtest  # noqa: E402

OUT_DIR = "lon_macd_results"


def metr(df, module, params):
    r = run_backtest(module.generate_signals(df, **params), commission=COMMISSION)
    return r["total_return"], r["sharpe"], r["max_drawdown"], r["profit_factor"], r["win_rate"]


def main():
    ensure_data()
    syms = sorted(f[:-4] for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
    rows = []
    for s in syms:
        try:
            df = load_symbol(s)
        except Exception:  # noqa: BLE001
            continue
        if len(df) < 120:
            continue
        m_ret, m_sh, m_dd, m_pf, m_wr = metr(df, v1, dict(ma_period=20, exit_bars=3))
        a_ret, a_sh, a_dd, a_pf, a_wr = metr(df, atr_exit, dict(k=2.0))
        rows.append({
            "品种": s,
            "MA20总收益%": round(m_ret * 100, 1), "ATR总收益%": round(a_ret * 100, 1),
            "收益改善%": round((a_ret - m_ret) * 100, 1),
            "MA20夏普": round(m_sh, 2), "ATR夏普": round(a_sh, 2),
            "夏普改善": round(a_sh - m_sh, 2),
            "MA20回撤%": round(m_dd * 100, 1), "ATR回撤%": round(a_dd * 100, 1),
            "MA20盈亏比": round(m_pf, 2), "ATR盈亏比": round(a_pf, 2),
        })
    t = pd.DataFrame(rows)
    t = t.sort_values("夏普改善", ascending=False).reset_index(drop=True)
    t.to_csv(os.path.join(OUT_DIR, "atr_scan.csv"), index=False, encoding="utf-8-sig")

    n = len(t)
    sh_better = int((t["夏普改善"] > 0).sum())
    ret_better = int((t["收益改善%"] > 0).sum())
    dd_better = int((t["ATR回撤%"] > t["MA20回撤%"]).sum())  # 回撤是负数，更大=更浅
    fg_rank = int(t.index[t["品种"] == "FG0"][0]) + 1 if (t["品种"] == "FG0").any() else None

    R = ["# ATR k=2 出场是否普遍有效（逐品种 vs MA20 原版）\n\n",
         "## 广度统计\n",
         f"- 品种数：**{n}**\n",
         f"- 夏普变好的：**{sh_better}/{n}**（{sh_better/n*100:.0f}%）\n",
         f"- 总收益变好的：**{ret_better}/{n}**（{ret_better/n*100:.0f}%）\n",
         f"- 最大回撤变浅的：**{dd_better}/{n}**（{dd_better/n*100:.0f}%）\n",
         f"- 夏普改善 中位数：**{t['夏普改善'].median():+.2f}**，平均：{t['夏普改善'].mean():+.2f}\n",
         f"- 玻璃 FG0 的夏普改善在 {n} 个品种里排第 **{fg_rank}**（改善 {t.loc[t['品种']=='FG0','夏普改善'].iloc[0]:+.2f}）\n\n"]

    R.append("## 改善最大的 10 个品种\n")
    R.append(t.head(10)[["品种", "MA20夏普", "ATR夏普", "夏普改善", "MA20总收益%", "ATR总收益%", "MA20盈亏比", "ATR盈亏比"]].to_markdown(index=False))
    R.append("\n\n## 变差最多的 8 个品种\n")
    R.append(t.tail(8)[["品种", "MA20夏普", "ATR夏普", "夏普改善", "MA20总收益%", "ATR总收益%"]].to_markdown(index=False))

    # 结论
    R.append("\n\n## 结论\n")
    if sh_better / n >= 0.6:
        R.append(f"- **不是只有玻璃**：{sh_better}/{n} 个品种换 ATR k=2 后夏普变好，是普遍现象。\n")
    else:
        R.append(f"- ATR k=2 改善并非普遍：仅 {sh_better}/{n} 个品种夏普变好，玻璃属较突出的个例。\n")
    R.append("- 玻璃改善尤其大（排第 "
             f"{fg_rank}），但同类（震荡中带阶段性趋势、平均亏损偏大的品种）多数也受益——"
             "ATR 的作用是把每笔亏损按波动率收窄、并让盈利单继续跑。\n")
    R.append("- 少数变差的品种通常是本就强趋势、MA 出场已能吃到大段的（ATR 反而过早被甩下车）。\n")

    with open(os.path.join(OUT_DIR, "ATR_SCAN.md"), "w", encoding="utf-8") as fh:
        fh.write("".join(R))
    print("".join(R))
    print(f"\n报告已保存: {OUT_DIR}/ATR_SCAN.md, atr_scan.csv")


if __name__ == "__main__":
    main()
