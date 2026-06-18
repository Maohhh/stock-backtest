"""
日内多周期 "看大做小" 演示：日线定方向 + 60 分钟择时
====================================================

把 "看大做小" 落到真正的日内尺度：

- **大周期**：用 *日线* EMA(20/60) 趋势判定方向（用完整历史，方向稳定）。
- **小周期**：在 *60 分钟* K 线上用均线交叉择时入场。
- 只做与日线方向一致的单（多空双向）。

⚠️ 数据局限：新浪分钟接口只返回最近约 1000 根 K 线，60 分钟约覆盖近 9 个月。
样本短，年化/夏普仅供框架演示与日内逻辑验证，**不能** 当作稳健的长期年化结论。
长期、可信的统计请看 ``run_futures_mtf.py``（21 年日线）。

用法：python scripts/run_intraday_mtf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.futures import FuturesDataSource, symbol_name
from src.backtest.futures_engine import FuturesBacktester, vol_target_weight

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

SYMBOLS = ["RB0", "HC0", "CU0", "AL0", "AU0", "M0", "TA0", "RU0"]
ANN_60M = 252 * 6          # 国内商品约 6 根 60 分钟/日（日盘+夜盘），近似年化
COST = 0.0005


def daily_bias(daily: pd.DataFrame, fast=20, slow=60) -> pd.DataFrame:
    """日线 EMA 趋势方向，返回 date->bias(-1/0/1)，已 shift 防未来函数。"""
    c = daily["close"]
    ef = c.ewm(span=fast, adjust=False).mean()
    es = c.ewm(span=slow, adjust=False).mean()
    bias = pd.Series(0.0, index=daily.index)
    bias[(ef > es) & (c > es) & (es.diff() > 0)] = 1.0
    bias[(ef < es) & (c < es) & (es.diff() < 0)] = -1.0
    out = pd.DataFrame({"d": pd.to_datetime(daily["date"]).dt.normalize(),
                        "bias": bias.shift(1).fillna(0.0)})
    return out


def intraday_signal(bars: pd.DataFrame, bias_df: pd.DataFrame,
                    fast=10, slow=30, mode="mtf") -> pd.Series:
    close = bars["close"].reset_index(drop=True)
    f = close.rolling(fast).mean()
    s = close.rolling(slow).mean()
    ltf = pd.Series(0.0, index=close.index)
    ltf[f > s] = 1.0
    ltf[f < s] = -1.0

    day = pd.to_datetime(bars["datetime"]).dt.normalize().reset_index(drop=True)
    bmap = bias_df.set_index("d")["bias"]
    bias = day.map(bmap).fillna(0.0)

    if mode == "ltf_only":
        sig = ltf
    elif mode == "htf_only":
        sig = bias
    else:
        sig = pd.Series(0.0, index=close.index)
        sig[(ltf > 0) & (bias.values > 0)] = 1.0
        sig[(ltf < 0) & (bias.values < 0)] = -1.0
    sig.index = bars["datetime"].values
    return sig


def backtest(bars, bias_df, mode):
    sig = intraday_signal(bars, bias_df, mode=mode)
    prices = bars.set_index("datetime")["close"]
    ret = prices.pct_change().clip(-0.1, 0.1).fillna(0.0)
    w = vol_target_weight(sig, ret, target_vol=0.15, vol_window=40,
                          max_leverage=3.0, ann=ANN_60M)
    bt = FuturesBacktester(cost=COST, ann=ANN_60M, ret_clip=0.1)
    return bt.run(prices, w)


def run():
    src = FuturesDataSource()
    rows = []
    for sym in SYMBOLS:
        daily = src.get_daily(sym)
        bars = src.get_minute(sym, 60)
        if daily is None or bars is None or len(bars) < 200:
            continue
        bias_df = daily_bias(daily)
        r_mtf = backtest(bars, bias_df, "mtf")
        r_ltf = backtest(bars, bias_df, "ltf_only")
        days = (pd.to_datetime(bars["datetime"]).max()
                - pd.to_datetime(bars["datetime"]).min()).days
        rows.append({
            "品种": sym, "名称": symbol_name(sym), "样本天数": days,
            "MTF期间收益": r_mtf.equity.iloc[-1] - 1, "MTF夏普": r_mtf["sharpe"],
            "MTF回撤": r_mtf["max_drawdown"],
            "仅60m期间收益": r_ltf.equity.iloc[-1] - 1, "仅60m夏普": r_ltf["sharpe"],
        })
    df = pd.DataFrame(rows)
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 200)

    disp = df.copy()
    for c in ["MTF期间收益", "MTF回撤", "仅60m期间收益"]:
        disp[c] = disp[c].map(lambda v: f"{v*100:.1f}%")
    for c in ["MTF夏普", "仅60m夏普"]:
        disp[c] = disp[c].map(lambda v: f"{v:.2f}")

    print("=" * 64)
    print("日内 看大做小：日线定方向 + 60分钟择时（样本约 9 个月，仅演示）")
    print("=" * 64)
    print(disp.to_string(index=False))
    print(f"\n均值 MTF 夏普 {df['MTF夏普'].mean():.2f} | 仅60m 夏普 {df['仅60m夏普'].mean():.2f}")
    df.to_csv(RESULTS / "intraday_mtf.csv", index=False)
    print(f"✅ 已保存 {RESULTS}/intraday_mtf.csv")


if __name__ == "__main__":
    run()
