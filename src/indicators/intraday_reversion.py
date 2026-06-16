"""
日内短线均值回归指标 (Intraday Trend-Aligned Reversion)

面向期货 15min(亦可 5min)短线交易的复合反转指标。

设计第一性原理
----------------
短周期价格的本质是订单流, 大部分时间在均值回归、少数时间在趋势。
因此 *高胜率* 几乎只能来自"在局部极值处接恐慌/抛贪婪"的均值回归, 而它唯一的
杀手是趋势日。本指标用三块结构把高胜率和有界尾部同时拿到:

1. 伸展/衰竭 (在哪 fade):
   - 价格相对 *会话锚定 VWAP* 的偏离, 以 ATR 为单位 (stretch);
   - 衰竭确认 = 假突破回收 (扫出新极值却收回前一区间) + 放量高潮。
     -> 对应人性: 止损扎堆被猎杀、追涨杀跌者在极值处情绪化成交。

2. 状态过滤 (何时允许 fade): Kaufman 效率系数 (ER)。
   ER 低 = 震荡(可 fade), ER 高 = 趋势(回避)。

3. 大周期趋势同向 (只 fade 噪声, 不逆大势): 仅在更高周期趋势方向上接回调
   (上升趋势只做多/买跌, 下降趋势只做空/卖反弹)。
   -> "顺大势、fade 小噪声" 是最高胜率的结构。

止损放在被扫的极值之外 (结构止损), 把"偶尔一次大亏"封死; 目标设在 VWAP
(均值磁铁), 因此单笔高胜率而尾部有界。

输入 DataFrame 需含列: open, high, low, close, volume; 时间戳可为索引或
名为 'datetime'/'date' 的列 (用于会话锚定 VWAP)。所有信号均只用当根及之前
的数据 (无未来函数)。
"""

import numpy as np
import pandas as pd

from .atr import atr as _atr


def _session_id(ts: pd.Series, session_gap_min: float) -> pd.Series:
    """按相邻 K 线时间间隔切分交易会话 (隔夜/夜盘切换处 VWAP 重锚)。"""
    gap = ts.diff().dt.total_seconds().div(60.0)
    new_session = (gap.isna()) | (gap > session_gap_min)
    return new_session.cumsum()


def _session_vwap(df: pd.DataFrame, session: pd.Series) -> pd.Series:
    """会话内累计成交量加权平均价 (典型价 = (H+L+C)/3)。"""
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    vol = df['volume'].clip(lower=0)
    tpv = (tp * vol).groupby(session).cumsum()
    cum_vol = vol.groupby(session).cumsum().replace(0, np.nan)
    return (tpv / cum_vol).fillna(tp)


def _efficiency_ratio(close: pd.Series, period: int) -> pd.Series:
    """Kaufman 效率系数: |净变动| / Σ|逐根变动|, 越低越震荡。"""
    direction = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(period).sum()
    return (direction / volatility.replace(0, np.nan)).clip(0, 1)


def intraday_reversion(
    df: pd.DataFrame,
    atr_period: int = 20,
    session_gap_min: float = 240.0,
    stretch_thr: float = 1.2,
    er_period: int = 10,
    er_max: float = 0.40,
    vol_ma: int = 20,
    vol_mult: float = 1.2,
    swing_lookback: int = 10,
    trend_period: int = 96,
    trend_slope_lb: int = 8,
    stop_buffer: float = 0.5,
    trend_align: bool = True,
) -> pd.DataFrame:
    """计算日内趋势同向均值回归信号。

    参数 (默认值面向 15min):
        atr_period:     ATR 周期 (止损/伸展的波动单位)
        session_gap_min: 间隔超过该分钟数视为新交易会话, VWAP 重锚 (默认 240)
        stretch_thr:    触发 fade 的最小伸展 (|close-vwap|/ATR), 默认 1.8
        er_period:      效率系数窗口
        er_max:         ER 上限, 低于则判为震荡(允许 fade)
        vol_ma:         成交量均线窗口
        vol_mult:       放量高潮倍数 (volume > vol_ma * vol_mult)
        swing_lookback: 判定新局部极值 / 结构止损所用回看根数
        trend_period:   大周期趋势 EMA 周期 (96≈15min 的约 1.5 个交易日)
        trend_slope_lb: 趋势斜率回看根数
        stop_buffer:    结构止损在极值外再留的 ATR 倍数
        trend_align:    True=只顺大周期趋势 fade (上升只做多/下降只做空)

    返回 DataFrame (与输入同索引), 主要列:
        vwap, atr, stretch, er, regime_range,
        trend_ema, trend_dir(+1/-1/0),
        sweep_high, sweep_low, vol_spike,
        exhaustion_up, exhaustion_down,
        long_signal, short_signal, signal(+1/-1/0), score,
        stop_long, target_long, stop_short, target_short
    """
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"缺少必需列: {col}")

    out = pd.DataFrame(index=df.index)

    # --- 时间戳 (会话锚定用) ---
    if 'datetime' in df.columns:
        ts = pd.to_datetime(df['datetime'])
    elif 'date' in df.columns:
        ts = pd.to_datetime(df['date'])
    elif isinstance(df.index, pd.DatetimeIndex):
        ts = pd.Series(df.index, index=df.index)
    else:
        # 无时间戳: 退化为单一会话 (VWAP 从头累计)
        ts = pd.Series(pd.date_range('2000-01-01', periods=len(df), freq='15min').values,
                       index=df.index)
    ts = pd.Series(pd.to_datetime(ts).values, index=df.index)
    session = _session_id(ts, session_gap_min)

    # --- 锚: VWAP 与波动: ATR ---
    out['vwap'] = _session_vwap(df, session)
    out['atr'] = _atr(df, period=atr_period)
    atr_safe = out['atr'].replace(0, np.nan)

    # --- 伸展 (以 ATR 为单位的 VWAP 偏离) ---
    # stretch: 收盘偏离(展示用); stretch_high/low: 插针极值偏离(触发用, 衡量真正被
    # fade 的那一下伸展幅度, 因为扫针 K 线收盘会收回, 收盘偏离会系统性低估)
    out['stretch'] = (df['close'] - out['vwap']) / atr_safe
    out['stretch_high'] = (df['high'] - out['vwap']) / atr_safe
    out['stretch_low'] = (df['low'] - out['vwap']) / atr_safe

    # --- 状态过滤: 效率系数 ---
    out['er'] = _efficiency_ratio(df['close'], er_period)
    out['regime_range'] = out['er'] < er_max

    # --- 大周期趋势方向 ---
    trend_ema = df['close'].ewm(span=trend_period, min_periods=trend_period, adjust=False).mean()
    out['trend_ema'] = trend_ema
    rising = trend_ema > trend_ema.shift(trend_slope_lb)
    falling = trend_ema < trend_ema.shift(trend_slope_lb)
    trend_dir = pd.Series(0, index=df.index)
    trend_dir[(df['close'] > trend_ema) & rising] = 1
    trend_dir[(df['close'] < trend_ema) & falling] = -1
    out['trend_dir'] = trend_dir

    # --- 衰竭: 假突破回收 + 放量 ---
    prior_high = df['high'].rolling(swing_lookback).max().shift(1)
    prior_low = df['low'].rolling(swing_lookback).min().shift(1)
    bearish = df['close'] < df['open']
    bullish = df['close'] > df['open']
    # 扫出新高却收回前高之下 = 多头陷阱(为做空准备)
    out['sweep_high'] = (df['high'] > prior_high) & (df['close'] < prior_high) & bearish
    # 扫破新低却收回前低之上 = 空头陷阱(为做多准备)
    out['sweep_low'] = (df['low'] < prior_low) & (df['close'] > prior_low) & bullish

    vol_ref = df['volume'].rolling(vol_ma).mean()
    out['vol_spike'] = df['volume'] > (vol_ref * vol_mult)

    # --- 衰竭信号 (插针极值伸展 + 放量确认) ---
    out['exhaustion_up'] = (
        out['sweep_high'] & (out['stretch_high'] >= stretch_thr) & out['vol_spike']
    )
    out['exhaustion_down'] = (
        out['sweep_low'] & (out['stretch_low'] <= -stretch_thr) & out['vol_spike']
    )

    # --- 合成信号: 震荡 + (可选)顺大周期趋势 ---
    long_ok = out['exhaustion_down'] & out['regime_range']
    short_ok = out['exhaustion_up'] & out['regime_range']
    if trend_align:
        long_ok = long_ok & (out['trend_dir'] == 1)
        short_ok = short_ok & (out['trend_dir'] == -1)
    else:
        long_ok = long_ok & (out['trend_dir'] >= 0)
        short_ok = short_ok & (out['trend_dir'] <= 0)

    out['long_signal'] = long_ok
    out['short_signal'] = short_ok
    out['signal'] = 0
    out.loc[long_ok, 'signal'] = 1
    out.loc[short_ok, 'signal'] = -1

    # --- 结构止损 / 均值目标 ---
    swing_low = df['low'].rolling(swing_lookback).min()
    swing_high = df['high'].rolling(swing_lookback).max()
    out['stop_long'] = swing_low - stop_buffer * out['atr']
    out['target_long'] = out['vwap']
    out['stop_short'] = swing_high + stop_buffer * out['atr']
    out['target_short'] = out['vwap']

    # --- 复合分数 (-100..100, 仅用于展示/排序): 越极端越倾向反转 ---
    raw = (-out['stretch'] / max(stretch_thr, 1e-9)).clip(-2, 2) * 50
    raw = raw.where(out['regime_range'], raw * 0.3)  # 趋势态衰减
    out['score'] = raw.clip(-100, 100)

    return out
