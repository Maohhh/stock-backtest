"""期货求魔交易系统 — 指标实现 (qiumo indicator).

A 5-minute futures swing & intraday trading model attributed to UP 「期货求魔」.

Conceptual summary
------------------
The system stacks three SMAs on one chart and treats them as a layered
support/resistance scaffold:

    MA20    — fast / exit reference
    MA120   — mid / partial-exit reference, "trend filter"
    MA250   — slow / regime anchor, last-line support / resistance

Regime is read off MA250 slope:
    slope(MA250) > eps   → uptrend (look for longs only)
    slope(MA250) < -eps  → downtrend (look for shorts only)
    |slope(MA250)| ≤ eps → range / stand aside

Long entry pattern near MA250 (mirror for shorts):
    signal_1 — double-bottom near MA250 then break the intervening high
    signal_2 — pulled back to MA250, then printed a higher swing low

Trailing exit ladder (long; mirror for shorts):
    1. |close - MA250| ranks in top `extreme_pct` over `extreme_window`
       bars → take full profit (spread is historically extended)
    2. close below MA120 → partial exit (most of the position)
    3. close below MA250 → full exit (regime anchor lost)

This module produces one feature DataFrame; strategies layer position
management on top.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _swing_low(low: pd.Series, k: int) -> pd.Series:
    """True at bars whose low is the strict min of [-k, +k] window."""
    rolling_min = low.rolling(2 * k + 1, center=True).min()
    return (low == rolling_min) & low.notna()


def _swing_high(high: pd.Series, k: int) -> pd.Series:
    rolling_max = high.rolling(2 * k + 1, center=True).max()
    return (high == rolling_max) & high.notna()


def qiumo(
    df: pd.DataFrame,
    fast: int = 20,
    mid: int = 120,
    slow: int = 250,
    slope_window: int = 10,
    slope_eps: Optional[float] = None,
    pivot_k: int = 3,
    near_tol: float = 0.003,
    extreme_window: int = 1200,
    extreme_pct: float = 0.95,
) -> pd.DataFrame:
    """Compute the 期货求魔 feature set on an OHLCV DataFrame.

    Parameters
    ----------
    df : DataFrame
        Must contain ``open, high, low, close`` columns. Index assumed to be
        time-ordered ascending. ``volume`` is unused but tolerated.
    fast, mid, slow : int
        SMA windows. Defaults follow the lecture: 20 / 120 / 250.
    slope_window : int
        Bars used to measure MA(slow) slope (fits a line over the last N bars).
    slope_eps : float or None
        Minimum |slope| / price to call a trend. ``None`` → auto, set to the
        median absolute slope over the series so roughly half the bars are
        flagged as trending. Tighten for cleaner regime tags.
    pivot_k : int
        Half-width of the pivot detection window. A swing low/high needs to be
        the strict min/max of ``2*pivot_k + 1`` bars centered on itself. With
        the default 3 this looks 3 bars left + 3 bars right.
    near_tol : float
        Relative distance from MA(slow) that still counts as "near". A pivot
        low is "near MA250" iff ``|low - MA250| / MA250 ≤ near_tol``.
        Default 0.3% suits futures 5-min bars; widen for noisier products.
    extreme_window, extreme_pct : int, float
        Spread-extreme detector. ``extreme_window`` 5-min bars ≈ the
        lookback the lecture suggests for "is this deviation extreme":
        1 week ≈ 1200, half month ≈ 2400, 1 month ≈ 4800, 1 year ≈ 12000.
        When ``|close - MA250|`` ranks ≥ ``extreme_pct`` percentile over
        that window, the system flags it as overextended (a take-profit
        trigger for an open position OR a mean-reversion entry candidate
        for a flat position — see ``QiuMoSwingStrategy``).

    Returns
    -------
    DataFrame indexed like ``df`` with columns:

    ma_fast, ma_mid, ma_slow
        The three SMAs.
    slope_slow
        MA(slow) slope per bar, normalized by price (dimensionless).
    regime
        +1 uptrend, -1 downtrend, 0 sideways.
    pivot_low, pivot_high
        Confirmed swing pivots (NaN elsewhere; values are the price).
    near_slow_low, near_slow_high
        Boolean: pivot low/high formed near MA(slow).
    signal1_long, signal2_long
        Double-bottom and higher-low pullback long triggers.
    signal1_short, signal2_short
        Mirror short triggers.
    spread
        ``close - MA(slow)``.
    spread_pct
        ``spread / MA(slow)``.
    spread_extreme_long, spread_extreme_short
        ``True`` when an open long/short is overextended (take-all-profit
        trigger).
    exit_long_loose, exit_long_tight
        ``True`` when a long should reduce (MA120 break) or fully exit
        (MA250 break). Mirror columns for shorts.
    """
    for col in ("high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"qiumo: missing column '{col}'")

    out = pd.DataFrame(index=df.index)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    out["ma_fast"] = close.rolling(fast, min_periods=fast).mean()
    out["ma_mid"] = close.rolling(mid, min_periods=mid).mean()
    out["ma_slow"] = close.rolling(slow, min_periods=slow).mean()

    # slope of MA(slow): fit y = a*x + b over last N points, take a / price
    # so the threshold is dimensionless.
    def _slope(arr: np.ndarray) -> float:
        if np.isnan(arr).any():
            return np.nan
        x = np.arange(arr.size, dtype=float)
        a, _ = np.polyfit(x, arr, 1)
        return a

    out["slope_slow"] = (
        out["ma_slow"].rolling(slope_window).apply(_slope, raw=True) / close
    )

    if slope_eps is None:
        slope_eps = float(out["slope_slow"].abs().median(skipna=True))
        if not np.isfinite(slope_eps):
            slope_eps = 0.0

    regime = pd.Series(0, index=df.index, dtype="int8")
    regime[out["slope_slow"] > slope_eps] = 1
    regime[out["slope_slow"] < -slope_eps] = -1
    out["regime"] = regime

    sl_mask = _swing_low(low, pivot_k)
    sh_mask = _swing_high(high, pivot_k)
    out["pivot_low"] = low.where(sl_mask)
    out["pivot_high"] = high.where(sh_mask)

    # "near MA(slow)" — pivot lows within near_tol of MA(slow).
    near_low = sl_mask & ((low - out["ma_slow"]).abs() / out["ma_slow"] <= near_tol)
    near_high = sh_mask & ((high - out["ma_slow"]).abs() / out["ma_slow"] <= near_tol)
    out["near_slow_low"] = near_low.fillna(False)
    out["near_slow_high"] = near_high.fillna(False)

    # signals are one-shot booleans (True only on the bar of trigger),
    # which is what a vector backtester expects.
    last_near_low = None
    prev_near_low = None
    interim_high = -np.inf
    sig1_long = np.zeros(len(df), dtype=bool)
    sig2_long = np.zeros(len(df), dtype=bool)
    sig1_short = np.zeros(len(df), dtype=bool)
    sig2_short = np.zeros(len(df), dtype=bool)

    last_near_high = None
    prev_near_high = None
    interim_low = np.inf

    sig1_long_armed = False
    sig2_long_armed = False
    sig1_short_armed = False
    sig2_short_armed = False

    close_v = close.values
    near_low_v = out["near_slow_low"].values
    near_high_v = out["near_slow_high"].values
    low_v = low.values
    high_v = high.values
    pivot_low_v = out["pivot_low"].values
    pivot_high_v = out["pivot_high"].values
    ma_slow_v = out["ma_slow"].values
    regime_v = regime.values

    for i in range(len(df)):
        if near_low_v[i]:
            prev_near_low = last_near_low
            last_near_low = (i, pivot_low_v[i])
            interim_high = -np.inf
            # arm long signals on each new near-MA(slow) pivot low
            if prev_near_low is not None:
                if last_near_low[1] >= prev_near_low[1]:
                    sig1_long_armed = True  # W-bottom candidate
                if last_near_low[1] > prev_near_low[1]:
                    sig2_long_armed = True  # higher-low candidate
        if last_near_low is not None and i > last_near_low[0]:
            interim_high = max(interim_high, high_v[i])

        if near_high_v[i]:
            prev_near_high = last_near_high
            last_near_high = (i, pivot_high_v[i])
            interim_low = np.inf
            if prev_near_high is not None:
                if last_near_high[1] <= prev_near_high[1]:
                    sig1_short_armed = True
                if last_near_high[1] < prev_near_high[1]:
                    sig2_short_armed = True

        # fire signal_1_long when the W-breakout actually triggers
        if (
            sig1_long_armed
            and np.isfinite(interim_high)
            and close_v[i] > interim_high
            and regime_v[i] >= 0
        ):
            sig1_long[i] = True
            sig1_long_armed = False

        # fire signal_2_long on first bar after HL where close re-asserts
        # above MA(slow) in confirmed uptrend.
        if (
            sig2_long_armed
            and close_v[i] > ma_slow_v[i]
            and regime_v[i] == 1
        ):
            sig2_long[i] = True
            sig2_long_armed = False

        if (
            sig1_short_armed
            and np.isfinite(interim_low)
            and close_v[i] < interim_low
            and regime_v[i] <= 0
        ):
            sig1_short[i] = True
            sig1_short_armed = False

        if (
            sig2_short_armed
            and close_v[i] < ma_slow_v[i]
            and regime_v[i] == -1
        ):
            sig2_short[i] = True
            sig2_short_armed = False

    out["signal1_long"] = sig1_long
    out["signal2_long"] = sig2_long
    out["signal1_short"] = sig1_short
    out["signal2_short"] = sig2_short

    out["spread"] = close - out["ma_slow"]
    out["spread_pct"] = out["spread"] / out["ma_slow"]

    abs_spread = out["spread"].abs()
    # rolling percentile rank: where today's |spread| sits within the last
    # extreme_window bars. 1.0 = current is the biggest in window.
    out["spread_rank"] = abs_spread.rolling(extreme_window, min_periods=extreme_window // 2).rank(pct=True)
    out["spread_extreme_long"] = (out["spread_rank"] >= extreme_pct) & (out["spread"] > 0)
    out["spread_extreme_short"] = (out["spread_rank"] >= extreme_pct) & (out["spread"] < 0)

    # Two-step exit ladder. "loose" = partial (MA120 break), "tight" = full
    # (MA250 break). Naming reflects the trader's intent: MA120 break is
    # the early warning, MA250 break is the line you don't tolerate.
    out["exit_long_loose"] = close < out["ma_mid"]
    out["exit_long_tight"] = close < out["ma_slow"]
    out["exit_short_loose"] = close > out["ma_mid"]
    out["exit_short_tight"] = close > out["ma_slow"]

    return out


__all__ = ["qiumo"]
