"""
单标的趋势指标 —— 端到端运行（可上同花顺主图）

对**单个品种**回测多周期趋势集成 + ER 过滤 + 波动率目标策略：
- 默认黄金 AU0（样本外最稳健）；可传任意品种代码
- 训练/样本外切分、杠杆档位、净值图

用法:
    python run_trend_indicator.py [品种代码] [周期]
    例: python run_trend_indicator.py AU0 daily
        python run_trend_indicator.py CU0 daily
"""

import os
import sys

import numpy as np
import pandas as pd

from src.data.futures import download_daily
from src.strategies.trend_following import TrendFollowingStrategy

DATA_DIR = "data/futures_daily"
COST = 0.0002          # 单边（黄金等大合约名义价值高，手续费+滑点≈2bp，且趋势换手低）
SPLIT = "2019-01-01"


def load(symbol: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        df = download_daily(symbol)
        if df is None:
            raise SystemExit(f"无法下载 {symbol} 日线数据")
        df.to_csv(path)
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    alias = {"open": "o", "high": "h", "low": "l", "close": "c"}
    out = {}
    for k, v in alias.items():
        out[k] = pd.to_numeric(df[k] if k in df.columns else df[v], errors="coerce")
    return pd.DataFrame(out).sort_index()


def show(res, name):
    print(f"  {name:14s} 夏普={res.sharpe:5.2f} 年化={res.ann_return*100:6.1f}% "
          f"最大回撤={res.max_drawdown*100:6.1f}% 年化波动={res.ann_vol*100:5.1f}%")


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "AU0"
    df = load(symbol)
    print("=" * 60)
    print(f"品种 {symbol} 日线 | {df.index[0].date()} ~ {df.index[-1].date()} | {len(df)} 天")
    print("=" * 60)

    strat = TrendFollowingStrategy(target_vol=0.30, cost=COST, bars_per_year=252)
    full = strat.run(df, symbol)
    print("\n[趋势指标 · 目标波动30%]")
    show(full, "全样本")
    show(strat.run(df.loc[:SPLIT], symbol), "训练<=2018")
    show(strat.run(df.loc[SPLIT:], symbol), "样本外>=2019")

    pos = strat.position(df)
    print(f"\n仓位分布: 多 {(pos>0.01).mean()*100:.0f}% / 空 {(pos<-0.01).mean()*100:.0f}% "
          f"/ 空仓 {(pos.abs()<=0.01).mean()*100:.0f}% | 年换手 ≈ "
          f"{pos.diff().abs().sum()/(len(pos)/252):.0f} 次")

    print("\n[年化-杠杆 前沿] 样本外>=2019（夏普不变，年化与回撤同向放大）")
    for tv in [0.20, 0.30, 0.40, 0.60]:
        r = TrendFollowingStrategy(target_vol=tv, cost=COST, max_leverage=6.0).run(df.loc[SPLIT:], symbol)
        print(f"  目标{tv*100:>3.0f}%vol -> 年化{r.ann_return*100:6.1f}% 回撤{r.max_drawdown*100:6.1f}% 夏普{r.sharpe:.2f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        oos = strat.run(df.loc[SPLIT:], symbol)
        fig, ax = plt.subplots(2, 1, figsize=(11, 8))
        full.equity.plot(ax=ax[0], color="navy",
                         label=f"Full ann={full.ann_return*100:.0f}% SR={full.sharpe:.2f} maxDD={full.max_drawdown*100:.0f}%")
        ax[0].axvline(pd.Timestamp(SPLIT), color="gray", ls="--")
        ax[0].set_title(f"{symbol} Trend-Ensemble (vol-target 30%) - equity (dashed = OOS start)")
        ax[0].legend(); ax[0].grid(alpha=.3); ax[0].set_yscale("log")
        oos.equity.plot(ax=ax[1], color="green",
                        label=f"OOS>=2019 ann={oos.ann_return*100:.0f}% SR={oos.sharpe:.2f} maxDD={oos.max_drawdown*100:.0f}%")
        ax[1].set_title("Out-of-sample equity"); ax[1].legend(); ax[1].grid(alpha=.3)
        plt.tight_layout()
        out = f"data/{symbol}_trend.png"
        plt.savefig(out, dpi=110)
        print(f"\n净值图已保存: {out}")
    except Exception as e:
        print(f"（绘图跳过: {e}）")


if __name__ == "__main__":
    main()
