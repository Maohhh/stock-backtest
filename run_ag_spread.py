"""
农产品配对价差回归指标 —— 端到端运行

默认玉米 C0 / 玉米淀粉 CS0（加工价差，高胜率）。也可传其他协整农产品配对。

用法:
    python run_ag_spread.py                 # 玉米-淀粉
    python run_ag_spread.py M0 RM0 ratio    # 豆粕-菜粕（比值价差）
"""

import os
import sys

import numpy as np
import pandas as pd

from src.data.futures import download_daily
from src.strategies.spread_reversion import PairReversionStrategy

DATA_DIR = "data/futures_daily"
SPLIT = "2020-07-01"


def load_clean(symbol: str) -> pd.Series:
    """读取（必要时下载）日线收盘价，并清洗连续主力拼接处的异常点。"""
    path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        df = download_daily(symbol)
        if df is None:
            raise SystemExit(f"无法下载 {symbol}")
        df.to_csv(path)
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    px = pd.to_numeric(df["close"] if "close" in df.columns else df["c"], errors="coerce")
    # 单点暴跳（相邻 >40%）视为坏数据，插值修复
    px[px.pct_change().abs() > 0.4] = np.nan
    return px.interpolate().rename(symbol)


def show(res, name):
    s = res.summary()
    print(f"  {name:14s} 笔数={s['交易笔数']:>3} 胜率={s['胜率']:>4}% "
          f"盈亏比={s['盈亏比']:>4} 盈利因子={s['盈利因子']:>4} 年化估={s['年化收益(估)']:>5}%")


def main():
    a = sys.argv[1] if len(sys.argv) > 1 else "C0"
    b = sys.argv[2] if len(sys.argv) > 2 else "CS0"
    mode = sys.argv[3] if len(sys.argv) > 3 else "diff"
    pa, pb = load_clean(a), load_clean(b)
    df = pd.concat([pa, pb], axis=1).dropna()
    print("=" * 64)
    print(f"农产品配对价差回归: {b} - {a}  ({mode})")
    print(f"区间 {df.index[0].date()} ~ {df.index[-1].date()} | {len(df)} 天")
    print("=" * 64)

    strat = PairReversionStrategy(window=30, entry=1.5, exit=0.3, stop=4.0,
                                  mode=mode, cost_points=8.0)
    full = strat.run(df[a], df[b])
    print("\n[价差回归 · 默认 n30/entry1.5σ/exit0.3σ/stop4σ]")
    show(full, "全样本")
    show(strat.run(df.loc[:SPLIT, a], df.loc[:SPLIT, b]), f"训练<{SPLIT}")
    show(strat.run(df.loc[SPLIT:, a], df.loc[SPLIT:, b]), "样本外>=")

    print("\n[胜率 / 盈亏比 取舍] 全样本（窗口越长入场越严→胜率越高，盈亏比略降）")
    for n, e in [(20, 1.5), (30, 1.5), (40, 1.0), (40, 1.5)]:
        r = PairReversionStrategy(window=n, entry=e, exit=0.3, stop=4.0, mode=mode).run(df[a], df[b])
        s = r.summary()
        print(f"  n{n}/entry{e}: 胜率={s['胜率']:>4}% 盈亏比={s['盈亏比']:>4} "
              f"盈利因子={s['盈利因子']:>4} 笔数={s['交易笔数']}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        z = strat.zscore(df[a], df[b])
        eq_dates = [t.exit_date for t in full.trades]
        eq = np.cumsum(full.pnl) * full.point_value
        fig, ax = plt.subplots(2, 1, figsize=(11, 8))
        ax[0].plot(z.index, z.values, color="purple", lw=0.8, label="spread z-score")
        for lv, c in [(1.5, "red"), (-1.5, "green"), (4, "gray"), (-4, "gray"), (0.3, "orange"), (-0.3, "orange")]:
            ax[0].axhline(lv, color=c, ls="--", lw=0.7)
        ax[0].set_title(f"{b}-{a} spread z-score (entry +/-1.5, exit +/-0.3, stop +/-4)")
        ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].plot(eq_dates, eq, color="darkgreen", drawstyle="steps-post",
                   label=f"win={full.win_rate*100:.0f}% payoff={full.payoff_ratio:.2f} PF={full.profit_factor:.2f}")
        ax[1].set_title("Cumulative PnL (yuan/lot-pair)"); ax[1].legend(); ax[1].grid(alpha=.3)
        plt.tight_layout()
        out = f"data/{a}_{b}_spread.png"
        plt.savefig(out, dpi=110)
        print(f"\n图已保存: {out}")
    except Exception as e:
        print(f"（绘图跳过: {e}）")


if __name__ == "__main__":
    main()
