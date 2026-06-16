"""
开盘区间突破指标 (Opening Range Breakout, ORB)

面向期货日内短线的*极简*顺势突破指标 —— 全部规则只有 4 条:

  1. 开盘区间: 每个交易会话(国内期货日盘/夜盘各算一个)的前 N 根 K 线划出高低区间;
  2. 突破: 收盘价突破区间上沿(做多)/下沿(做空);
  3. 方向过滤: 突破方向须与会话 VWAP 一致(站上 VWAP 才做多, 跌破才做空);
  4. 放量过滤: 突破那根的相对成交量(RVOL)达标 —— 只在"有行情(in play)"时出手。

止损放在开盘区间的另一侧, 利润让它奔跑(由回测的会话末/跟踪止损了结)。

为什么是突破而不是高胜率反转
-----------------------------
大样本实证(Zarattini/Barbon/Aziz, 2016-2023, 7000+股)显示: 受限于"当日放量异动"
标的的 5 分钟开盘区间突破有显著正期望(夏普~2.8)。它是*低胜率高盈亏比*的顺势结构 ——
靠少数趋势日的大赢覆盖多数小亏, 与"让利润奔跑"一致。简单、规则少、不易过拟合, 正是
能稳定执行的短线策略的共性。

输入 DataFrame 需含: open, high, low, close, volume; 时间戳为索引或 'datetime'/'date' 列。
所有信号只用当根及之前数据(无未来函数)。
"""

import numpy as np
import pandas as pd

from .atr import atr as _atr
from .intraday_reversion import _session_id, _session_vwap


def opening_range_breakout(
    df: pd.DataFrame,
    or_bars: int = 2,
    session_gap_min: float = 240.0,
    rvol_period: int = 20,
    rvol_min: float = 1.2,
    vwap_filter: bool = True,
    one_per_session: bool = True,
    atr_period: int = 20,
) -> pd.DataFrame:
    """计算开盘区间突破信号。

    参数(默认面向 15min):
        or_bars:         开盘区间由每会话前几根 K 线确定(2≈30min)
        session_gap_min: 间隔超过该分钟数视为新会话(日盘/夜盘分开)
        rvol_period:     相对成交量的基准窗口(滚动中位数)
        rvol_min:        突破那根 RVOL 下限(只做放量突破)
        vwap_filter:     是否要求突破方向与会话 VWAP 一致
        one_per_session: 每会话每方向只取首次突破
        atr_period:      ATR 周期(供回测做跟踪止损)

    返回 DataFrame, 主要列:
        session, bar_in_session, vwap, atr, or_high, or_low, rvol,
        long_signal, short_signal, signal(+1/-1/0),
        stop_long(=or_low), stop_short(=or_high)
    """
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"缺少必需列: {col}")

    out = pd.DataFrame(index=df.index)

    # 时间戳 -> 会话
    if 'datetime' in df.columns:
        ts = pd.to_datetime(df['datetime'])
    elif 'date' in df.columns:
        ts = pd.to_datetime(df['date'])
    elif isinstance(df.index, pd.DatetimeIndex):
        ts = pd.Series(df.index, index=df.index)
    else:
        ts = pd.Series(pd.date_range('2000-01-01', periods=len(df), freq='15min').values,
                       index=df.index)
    ts = pd.Series(pd.to_datetime(ts).values, index=df.index)
    session = _session_id(ts, session_gap_min)
    out['session'] = session
    out['vwap'] = _session_vwap(df, session)
    out['atr'] = _atr(df, period=atr_period)

    # 会话内序号 + 开盘区间高低(前 or_bars 根的 max/min, 广播到整个会话)
    bar_in_session = session.groupby(session).cumcount()
    out['bar_in_session'] = bar_in_session
    is_or = bar_in_session < or_bars
    out['or_high'] = df['high'].where(is_or).groupby(session).transform('max')
    out['or_low'] = df['low'].where(is_or).groupby(session).transform('min')

    # 相对成交量(对滚动中位数), 对放量更稳健
    vol_ref = df['volume'].rolling(rvol_period, min_periods=rvol_period // 2).median()
    out['rvol'] = df['volume'] / vol_ref.replace(0, np.nan)

    active = bar_in_session >= or_bars  # 开盘区间形成后才允许突破
    vol_ok = out['rvol'] >= rvol_min

    long_raw = active & (df['close'] > out['or_high']) & vol_ok
    short_raw = active & (df['close'] < out['or_low']) & vol_ok
    if vwap_filter:
        long_raw = long_raw & (df['close'] > out['vwap'])
        short_raw = short_raw & (df['close'] < out['vwap'])

    if one_per_session:
        long_raw = long_raw & (long_raw.groupby(session).cumsum() == 1)
        short_raw = short_raw & (short_raw.groupby(session).cumsum() == 1)

    out['long_signal'] = long_raw.fillna(False)
    out['short_signal'] = short_raw.fillna(False)
    out['signal'] = 0
    out.loc[out['long_signal'], 'signal'] = 1
    out.loc[out['short_signal'], 'signal'] = -1

    out['stop_long'] = out['or_low']    # 止损放区间另一侧
    out['stop_short'] = out['or_high']

    return out
