"""期货求魔交易系统 — 策略实现。

Two pre-built strategies layered on top of ``src.indicators.qiumo``:

    QiuMoSwingStrategy    — 5-min swing model.
        Entry  : signal1/signal2 long/short near MA250.
        Exit   : ladder via spread extreme → MA120 break (partial) → MA250
                 break (full).
    QiuMoDayStrategy      — 5-min / 3-min day-trade model.
        Entry  : pullback to MA120 with regime confirmation by MA250.
        Exit   : close on the opposite side of MA20, or end-of-session.

Both strategies are self-contained — they don't depend on a framework base
class. Each exposes ``generate_signals(df) -> pd.DataFrame`` which is the
cleanest input for a vector backtester. ``run_backtest`` is a tiny helper
that turns those signals into equity / PnL series.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.indicators.qiumo import qiumo


@dataclass
class QiuMoSwingStrategy:
    """5-minute swing model from the 求魔 lectures.

    Position rules:
        - Open long on signal1_long or signal2_long (only if flat).
        - Open short on signal1_short or signal2_short (only if flat).
        - Exit long if spread_extreme_long OR exit_long_tight (close < MA250).
          Partial exit on exit_long_loose (close < MA120) is approximated
          here by full exit since vector backtests don't model partial
          position scaling cleanly; override ``partial_exit_pct`` if you
          want to half-out instead.
        - Mirror for shorts.

    A position is held at most ``max_hold_bars`` bars (None = unlimited).
    """

    fast: int = 20
    mid: int = 120
    slow: int = 250
    slope_window: int = 10
    slope_eps: Optional[float] = None
    pivot_k: int = 3
    near_tol: float = 0.003
    extreme_window: int = 240
    extreme_pct: float = 0.95
    use_partial_exit: bool = True
    max_hold_bars: Optional[int] = None

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        feat = qiumo(
            df,
            fast=self.fast,
            mid=self.mid,
            slow=self.slow,
            slope_window=self.slope_window,
            slope_eps=self.slope_eps,
            pivot_k=self.pivot_k,
            near_tol=self.near_tol,
            extreme_window=self.extreme_window,
            extreme_pct=self.extreme_pct,
        )

        pos = np.zeros(len(df), dtype=np.int8)
        entry_idx = -1
        cur = 0
        for i in range(len(df)):
            new = cur
            if cur == 0:
                if feat["signal1_long"].iat[i] or feat["signal2_long"].iat[i]:
                    new = 1
                    entry_idx = i
                elif feat["signal1_short"].iat[i] or feat["signal2_short"].iat[i]:
                    new = -1
                    entry_idx = i
            elif cur == 1:
                tight = bool(feat["exit_long_tight"].iat[i])
                extreme = bool(feat["spread_extreme_long"].iat[i])
                loose = bool(feat["exit_long_loose"].iat[i]) and self.use_partial_exit
                aged = self.max_hold_bars is not None and (i - entry_idx) >= self.max_hold_bars
                if tight or extreme or loose or aged:
                    new = 0
            elif cur == -1:
                tight = bool(feat["exit_short_tight"].iat[i])
                extreme = bool(feat["spread_extreme_short"].iat[i])
                loose = bool(feat["exit_short_loose"].iat[i]) and self.use_partial_exit
                aged = self.max_hold_bars is not None and (i - entry_idx) >= self.max_hold_bars
                if tight or extreme or loose or aged:
                    new = 0
            cur = new
            pos[i] = cur
            if cur == 0:
                entry_idx = -1

        out = feat.copy()
        out["position"] = pos
        return out


@dataclass
class QiuMoDayStrategy:
    """3-min / 5-min day-trade model.

    On a 3-min chart MA20 alone is the entry/exit reference. On a 5-min
    chart MA120 and MA250 are the entry references and MA20 is the exit.
    This class implements the 5-min variant by default; pass smaller
    windows (e.g. ``fast=5, mid=20, slow=120``) to approximate the 3-min
    flavor on 5-min bars.

    Entry: close pulls back to MA(mid), then closes back through MA(mid)
           in the direction of MA(slow) slope.
    Exit : close crosses MA(fast) against the position, OR session_close
           index (caller-provided boolean series) is True.
    """

    fast: int = 20
    mid: int = 120
    slow: int = 250
    slope_window: int = 10
    slope_eps: Optional[float] = None
    near_tol: float = 0.004
    require_regime: bool = True

    def generate_signals(
        self,
        df: pd.DataFrame,
        session_close: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        feat = qiumo(
            df,
            fast=self.fast,
            mid=self.mid,
            slow=self.slow,
            slope_window=self.slope_window,
            slope_eps=self.slope_eps,
            near_tol=self.near_tol,
        )
        close = df["close"].astype(float)
        ma_fast = feat["ma_fast"]
        ma_mid = feat["ma_mid"]
        regime = feat["regime"]

        # crossed-up / crossed-down detection
        prev_close = close.shift(1)
        crossed_up_mid = (prev_close <= ma_mid.shift(1)) & (close > ma_mid)
        crossed_dn_mid = (prev_close >= ma_mid.shift(1)) & (close < ma_mid)
        crossed_up_fast = (prev_close <= ma_fast.shift(1)) & (close > ma_fast)
        crossed_dn_fast = (prev_close >= ma_fast.shift(1)) & (close < ma_fast)

        # recent pullback to MA(mid): low touched within near_tol within
        # last 6 bars (covers a small consolidation against MA120).
        pulled_long = (
            (df["low"] - ma_mid).abs() / ma_mid <= self.near_tol
        ).rolling(6, min_periods=1).max().astype(bool)
        pulled_short = pulled_long  # symmetric tag

        long_entry = crossed_up_mid & pulled_long
        short_entry = crossed_dn_mid & pulled_short
        if self.require_regime:
            long_entry &= regime >= 0
            short_entry &= regime <= 0

        if session_close is None:
            session_close = pd.Series(False, index=df.index)

        pos = np.zeros(len(df), dtype=np.int8)
        cur = 0
        for i in range(len(df)):
            new = cur
            if session_close.iat[i] and cur != 0:
                new = 0
            elif cur == 0:
                if bool(long_entry.iat[i]):
                    new = 1
                elif bool(short_entry.iat[i]):
                    new = -1
            elif cur == 1 and bool(crossed_dn_fast.iat[i]):
                new = 0
            elif cur == -1 and bool(crossed_up_fast.iat[i]):
                new = 0
            cur = new
            pos[i] = cur

        out = feat.copy()
        out["long_entry"] = long_entry
        out["short_entry"] = short_entry
        out["position"] = pos
        return out


def run_backtest(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    cost_bps: float = 1.0,
) -> pd.DataFrame:
    """Vector backtest: apply ``signals['position']`` (shifted by 1 to avoid
    look-ahead) to bar-to-bar log returns. Costs are charged on every
    position change in basis points of notional.

    Returns
    -------
    DataFrame with columns: close, position, bar_ret, strat_ret, equity,
    drawdown, trade_count.
    """
    close = df["close"].astype(float)
    pos = signals["position"].astype(float).reindex(close.index).fillna(0)
    pos = pos.shift(1).fillna(0)  # execute next bar

    bar_ret = np.log(close).diff().fillna(0)
    turn = pos.diff().abs().fillna(0)
    cost = turn * (cost_bps / 10_000)
    strat_ret = pos * bar_ret - cost

    equity = strat_ret.cumsum().pipe(np.exp)
    drawdown = equity / equity.cummax() - 1
    return pd.DataFrame(
        {
            "close": close,
            "position": pos,
            "bar_ret": bar_ret,
            "strat_ret": strat_ret,
            "equity": equity,
            "drawdown": drawdown,
            "trade_count": turn.cumsum(),
        }
    )


__all__ = ["QiuMoSwingStrategy", "QiuMoDayStrategy", "run_backtest"]
