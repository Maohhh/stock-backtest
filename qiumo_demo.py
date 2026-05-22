"""期货求魔 demo: synthetic 5-min data → indicator → strategy → backtest.

Replace the synthetic generator with your own OHLCV DataFrame to run on
real futures data. Expected columns: open, high, low, close, volume;
index must be time-ordered ascending.

Run:
    pip install pandas numpy
    python qiumo_demo.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.qiumo import qiumo
from src.strategies.qiumo_strategy import (
    QiuMoDayStrategy,
    QiuMoSwingStrategy,
    run_backtest,
)


def make_demo_data(n: int = 5000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # mild trend with regime shifts to give the slope filter something to do
    drift = np.concatenate([
        np.full(n // 3, 0.0003),
        np.full(n // 3, -0.0002),
        np.full(n - 2 * (n // 3), 0.0001),
    ])
    sigma = 0.0035
    ret = rng.normal(drift, sigma)
    close = 1000 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, sigma / 2, n)))
    low = close * (1 - np.abs(rng.normal(0, sigma / 2, n)))
    op = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {"open": op, "high": high, "low": low, "close": close,
         "volume": rng.integers(100, 1000, n)},
        index=pd.date_range("2024-01-01 09:00", periods=n, freq="5min"),
    )


def summarize(name: str, bt: pd.DataFrame) -> None:
    equity_end = float(bt["equity"].iloc[-1])
    total_return = (equity_end - 1.0) * 100
    max_dd = float(bt["drawdown"].min()) * 100
    n_trades = int(bt["trade_count"].iloc[-1])
    # annualization assumes 5-min bars, 240 bars/day, 250 days/year
    bars_per_year = 240 * 250
    n_bars = len(bt.index)
    if n_bars and bt["strat_ret"].std() > 0:
        sharpe = (
            bt["strat_ret"].mean() / bt["strat_ret"].std()
            * np.sqrt(bars_per_year)
        )
    else:
        sharpe = float("nan")
    print(
        f"  {name:>6}  return={total_return:+7.2f}%  "
        f"max_dd={max_dd:7.2f}%  sharpe={sharpe:+.2f}  trades={n_trades}"
    )


def main() -> None:
    df = make_demo_data()

    feat = qiumo(df)
    print("=== feature snapshot ===")
    print(feat.tail(3))
    print()
    print(f"signal1_long fired {int(feat['signal1_long'].sum())} times")
    print(f"signal2_long fired {int(feat['signal2_long'].sum())} times")
    print(f"signal1_short fired {int(feat['signal1_short'].sum())} times")
    print(f"signal2_short fired {int(feat['signal2_short'].sum())} times")
    print(f"regime counts: {feat['regime'].value_counts().to_dict()}")
    print()

    print("=== backtest (synthetic 5-min, 5000 bars, 1bp cost) ===")
    for name, strat in [("swing", QiuMoSwingStrategy()), ("day", QiuMoDayStrategy())]:
        sig = strat.generate_signals(df)
        bt = run_backtest(df, sig, cost_bps=1.0)
        summarize(name, bt)


if __name__ == "__main__":
    main()
