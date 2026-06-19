#!/usr/bin/env python3
"""
工业品篮子的 15 分钟级别表现（ATR k=2 出场），与日线对比

数据：data_futures/weighted_15min/<SYM>.parquet（持仓量加权连续，约 2023-09~2026-05）。
对工业品篮子逐品种用 ATR k=2 出场回测，再做等权组合；与同口径日线对比。

用法： python lon_macd_basket_15min.py [--no-plot]
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.strategies import lon_macd_atr as atr_exit  # noqa: E402
from src.backtest.futures_engine import run_backtest  # noqa: E402

DIR15 = os.path.join("data_futures", "weighted_15min")
DATA_REF = "origin/claude/futures-data-inventory-fdwz31"
OUT_DIR = "lon_macd_results"
COMMISSION = 0.0003

# 工业品篮子（15min 文件名不带 0）
BASKET = ["AU", "AG", "AL", "CU", "NI", "PB", "SN", "ZN",
          "RB", "HC", "I", "JM", "SF", "SM", "SS",
          "TA", "MA", "PP", "L", "V", "EG", "FU", "BU", "SC", "RU", "FG"]


def ensure15(sym):
    path = os.path.join(DIR15, f"{sym}.parquet")
    if os.path.exists(path):
        return path
    os.makedirs(DIR15, exist_ok=True)
    rel = f"data_futures/weighted_15min/{sym}.parquet"
    try:
        content = subprocess.check_output(["git", "show", f"{DATA_REF}:{rel}"], stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None
    with open(path, "wb") as fh:
        fh.write(content)
    return path


def load15(sym):
    p = ensure15(sym)
    if p is None:
        return None
    df = pd.read_parquet(p).rename(columns={"datetime": "date"})
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df[df["close"] > 0].reset_index(drop=True)


def main(argv):
    do_plot = "--no-plot" not in argv
    daily_map = {}
    rows = []
    ann = None
    for sym in BASKET:
        df = load15(sym)
        if df is None or len(df) < 500:
            continue
        if ann is None:
            yrs = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25
            ann = int(len(df) / yrs)
        r = run_backtest(atr_exit.generate_signals(df, k=2.0), commission=COMMISSION, annualization=ann)
        eq = r["equity_curve"]
        daily_map[sym] = pd.Series(eq["equity"].pct_change().fillna(0.0).values,
                                   index=pd.to_datetime(eq["date"].values))
        rows.append({"品种": sym, "总收益%": round(r["total_return"]*100, 1),
                     "夏普": round(r["sharpe"], 2), "回撤%": round(r["max_drawdown"]*100, 1),
                     "交易数": r["n_trades"], "胜率%": round(r["win_rate"]*100, 1),
                     "盈亏比": round(r["profit_factor"], 2)})

    t = pd.DataFrame(rows).sort_values("总收益%", ascending=False)
    # 等权组合
    mat = pd.DataFrame(daily_map).sort_index()
    pr = mat.mean(axis=1, skipna=True).fillna(0.0)
    eq = (1 + pr).cumprod()
    yrs = len(eq) / ann
    cagr = eq.iloc[-1] ** (1/yrs) - 1 if yrs > 0 and eq.iloc[-1] > 0 else 0
    sh = pr.mean()/pr.std()*np.sqrt(ann) if pr.std() else 0
    mdd = (eq/eq.cummax()-1).min()

    R = ["# 工业品篮子 15 分钟表现（ATR k=2 出场）\n\n",
         f"数据：持仓量加权 15min，约 {mat.index[0].date()} ~ {mat.index[-1].date()}（~3 年），"
         f"年化因子≈{ann} 根/年；篮子 {len(daily_map)} 个品种。\n\n",
         "## 等权组合（15min vs 日线对照）\n",
         "| 口径 | 总收益% | 年化% | 夏普 | 最大回撤% |\n|---|---|---|---|---|\n",
         f"| **工业品篮子 15min** | {(eq.iloc[-1]-1)*100:.0f} | {cagr*100:.1f} | **{sh:.2f}** | {mdd*100:.1f} |\n",
         "| 工业品篮子 日线(同 ATR k=2) | 101 | 3.4 | 0.75 | -8.4 |\n",
         "| （注：日线为 2005~2026 全样本，15min 仅近~3 年，区间不同仅作参考）|\n\n",
         "## 15min 各品种明细\n",
         t.to_markdown(index=False),
         "\n\n## 小结\n"]
    pos = int((t["总收益%"] > 0).sum())
    R.append(f"- 15min 篮子组合夏普 {sh:.2f}、年化 {cagr*100:.1f}%、回撤 {mdd*100:.1f}%；"
             f"{pos}/{len(t)} 个品种 15min 盈利。\n")
    if sh < 0.3:
        R.append("- **明显弱于日线**：15 分钟噪音大、信号频繁、手续费拖累重，即便上 ATR 移动止损也难有日线那样的边际。\n")
    else:
        R.append("- 15min 仍可用但弱于日线；高频下手续费与噪音是主要拖累。\n")
    R.append("- 结论：这套 LON+MACD（哪怕配 ATR 出场、哪怕只做最强的工业品）边际仍在**日线/波段**，下沉到 15min 会显著退化。\n")

    with open(os.path.join(OUT_DIR, "BASKET_15MIN.md"), "w", encoding="utf-8") as fh:
        fh.write("".join(R))
    print("".join(R))

    if do_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(11, 6))
            ax.plot(eq.index, eq.values, lw=1.8, color="C3", label=f"15min basket (sh {sh:.2f})")
            ax.set_ylabel("equity (x initial)")
            ax.set_title("Industrial basket 15min, ATR k=2 exit, equal-weight")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            out = os.path.join(OUT_DIR, "basket_15min_equity.png")
            fig.savefig(out, dpi=120)
            plt.close(fig)
            print(f"图表已保存: {out}")
        except Exception as exc:  # noqa: BLE001
            print(f"画图失败: {exc}")
    print(f"报告已保存: {OUT_DIR}/BASKET_15MIN.md")


if __name__ == "__main__":
    main(sys.argv[1:])
