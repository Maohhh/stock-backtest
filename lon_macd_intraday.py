#!/usr/bin/env python3
"""
LON + MACD 趋势策略 —— 15 分钟（持仓量加权连续）日内回测

数据：data_futures/weighted_15min/<SYM>.parquet（持仓量加权连续，换月不跳空，
约 2023-09 ~ 2026-05）。缺失时自动从数据分支提取。

注意：所有周期参数按「根数」计（15min 一根）。MA20=20 根≈5 小时、连续 3 根≈45 分钟，
与日线语义不同。夏普按数据实际「每年根数」折算。手续费按每次换仓的名义额比例扣。

用法：
    python lon_macd_intraday.py            # 默认玻璃 FG
    python lon_macd_intraday.py RB CU      # 指定品种
    python lon_macd_intraday.py FG --no-plot
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.strategies.lon_macd_strategy import generate_signals  # noqa: E402
from src.backtest.futures_engine import run_backtest  # noqa: E402

DATA_DIR = os.path.join("data_futures", "weighted_15min")
DATA_REF = "origin/claude/futures-data-inventory-fdwz31"
OUT_DIR = "lon_macd_results"
COMMISSION = 0.0003


def ensure_symbol(sym: str) -> str:
    path = os.path.join(DATA_DIR, f"{sym}.parquet")
    if os.path.exists(path):
        return path
    os.makedirs(DATA_DIR, exist_ok=True)
    rel = f"data_futures/weighted_15min/{sym}.parquet"
    print(f"提取 {rel} ...")
    content = subprocess.check_output(["git", "show", f"{DATA_REF}:{rel}"])
    with open(path, "wb") as fh:
        fh.write(content)
    return path


def load_15min(sym: str) -> pd.DataFrame:
    df = pd.read_parquet(ensure_symbol(sym))
    df = df.rename(columns={"datetime": "date"})
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["close"] > 0].reset_index(drop=True)
    return df


def bars_per_year(df: pd.DataFrame) -> float:
    years = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25
    return len(df) / years if years > 0 else 252.0


def run_one(df: pd.DataFrame, ann: float, tag: str, **params) -> dict:
    sig = generate_signals(df, **params)
    r = run_backtest(sig, commission=COMMISSION, annualization=int(ann))
    print(f"=== 15min v1 [{tag}] ===")
    print(f"  总收益   : {r['total_return']*100:7.1f}%   (买入持有 {r['bh_return']*100:.1f}%)")
    print(f"  年化     : {r['cagr']*100:7.1f}%")
    print(f"  夏普     : {r['sharpe']:7.2f}   索提诺 {r['sortino']:.2f}  卡玛 {r['calmar']:.2f}")
    print(f"  最大回撤 : {r['max_drawdown']*100:7.1f}%")
    print(f"  交易次数 : {r['n_trades']}  (多 {r['long_trades']} / 空 {r['short_trades']})")
    print(f"  胜率     : {r['win_rate']*100:.1f}%   盈亏比 {r['profit_factor']:.2f}")
    print(f"  平均持仓 : {r['avg_hold_bars']:.1f} 根(15min)≈{r['avg_hold_bars']*15/60:.1f}小时   持仓占比 {r['exposure']*100:.1f}%")
    print()
    return r


def main(argv):
    do_plot = "--no-plot" not in argv
    syms = [a for a in argv if not a.startswith("--")] or ["FG"]
    os.makedirs(OUT_DIR, exist_ok=True)

    for sym in syms:
        df = load_15min(sym)
        ann = bars_per_year(df)
        print(f"\n品种 {sym}（15min 加权连续）: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
        print(f"共 {len(df)} 根, 年化因子≈{ann:.0f} 根/年\n")

        configs = [
            ("MA20,exit3", dict(ma_period=20, exit_bars=3)),
            ("MA10,exit3", dict(ma_period=10, exit_bars=3)),
            ("MA20,exit6", dict(ma_period=20, exit_bars=6)),
            ("MA460(~20d),exit3", dict(ma_period=460, exit_bars=3, warmup=500)),
        ]
        results = {}
        for tag, p in configs:
            results[tag] = run_one(df, ann, tag, **p)

        # 保存默认参数逐笔交易
        base = results["MA20,exit3"]["trades"].copy()
        if len(base):
            base["pnl%"] = (base["pnl_pct"] * 100).round(2)
            base.to_csv(os.path.join(OUT_DIR, f"{sym}_15min_trades_MA20.csv"),
                        index=False, encoding="utf-8-sig")

        if do_plot:
            try:
                _plot(sym, df, results)
            except Exception as exc:  # noqa: BLE001
                print(f"画图失败（可忽略）: {exc}")


def _plot(sym, df, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7))
    for tag, r in results.items():
        eq = r["equity_curve"]
        ax.plot(eq["date"], eq["equity"], lw=1.3, label=f"{tag} ({r['total_return']*100:.0f}%)")
    # 买入持有
    eq0 = list(results.values())[0]["equity_curve"]
    ax.plot(eq0["date"], eq0["bh_equity"], lw=1.2, color="gray", ls="--", label="buy&hold")
    ax.set_title(f"{sym} 15min LON+MACD v1 - param variants")
    ax.set_ylabel("equity (x initial)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"{sym}_15min_equity.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"图表已保存: {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
