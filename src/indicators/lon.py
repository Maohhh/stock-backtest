"""
LON 龙系长线指标 (Long-line Energy Indicator)

这是通达信经典的"龙系长线"指标，本质上是基于量价的能量潮(OBV类)
做 MACD 式的双指数平滑，用于判断中长期资金能量的强弱与拐点。

通达信公式：
    VID  := SUM(VOL,2)/((HHV(HIGH,2)-LLV(LOW,2))*100);
    RC   := (CLOSE-REF(CLOSE,1))*VID;
    LONG := SUM(RC,0);              # 0 表示从第一根K线起累积求和
    DIFF := SMA(LONG,10,1);
    DEA  := SMA(LONG,20,1);
    LON  : DIFF-DEA, COLORSTICK;    # 柱状能量（多头红柱 / 空头绿柱）
    LONMA: SMA(LON,6,1);            # LON 的均线

其中 SMA(X,N,M) 为通达信递归加权平均：
    Y_today = (M*X_today + (N-M)*Y_yesterday) / N
当 M=1 时等价于 alpha=1/N 的指数移动平均（EMA，不复权）。

LON 在 0 轴上方为多头能量(红柱)，下方为空头能量(绿柱)。
"LON长度" 指柱体长度，即 abs(LON)；柱体连续变短代表当前方向的能量在衰减。
"""

import pandas as pd
import numpy as np


def _tdx_sma(series: pd.Series, n: int, m: int = 1) -> pd.Series:
    """通达信 SMA(X,N,M) 递归加权平均，等价于 alpha=m/n 的 EMA。"""
    alpha = m / n
    return series.ewm(alpha=alpha, adjust=False, min_periods=1).mean()


def lon(df: pd.DataFrame,
        diff_period: int = 10,
        dea_period: int = 20,
        ma_period: int = 6) -> pd.DataFrame:
    """
    计算 LON 龙系长线指标

    参数:
        df: 包含 high, low, close, volume 列的 DataFrame
        diff_period: DIFF 平滑周期，默认 10
        dea_period:  DEA  平滑周期，默认 20
        ma_period:   LONMA 平滑周期，默认 6

    返回:
        DataFrame，包含以下列：
        - long:  LONG 量价能量累积值
        - diff:  LONG 的快线
        - dea:   LONG 的慢线
        - lon:   LON 柱状能量 (DIFF - DEA)；>0 多头能量，<0 空头能量
        - lonma: LON 的均线
        - lon_len: LON 柱体长度 abs(lon)

    示例:
        >>> res = lon(df)
        >>> res['lon']      # 长线能量柱
        >>> res['lon_len']  # 柱体长度
    """
    required = ['high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"缺少必需列: {col}")

    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)
    vol = df['volume'].astype(float)

    # VID = SUM(VOL,2) / ((HHV(HIGH,2)-LLV(LOW,2))*100)
    sum_vol2 = vol.rolling(window=2, min_periods=1).sum()
    hhv2 = high.rolling(window=2, min_periods=1).max()
    llv2 = low.rolling(window=2, min_periods=1).min()
    rng = (hhv2 - llv2) * 100
    # 防止区间为 0（横盘/一字板）导致除零
    vid = sum_vol2 / rng.replace(0, np.nan)
    vid = vid.replace([np.inf, -np.inf], np.nan).fillna(0)

    # RC = (CLOSE - REF(CLOSE,1)) * VID
    rc = close.diff().fillna(0) * vid

    # LONG = SUM(RC,0) 从头累积
    long = rc.cumsum()

    # DIFF / DEA / LON / LONMA
    diff = _tdx_sma(long, diff_period, 1)
    dea = _tdx_sma(long, dea_period, 1)
    lon_hist = diff - dea
    lonma = _tdx_sma(lon_hist, ma_period, 1)

    result = pd.DataFrame(index=df.index)
    result['long'] = long
    result['diff'] = diff
    result['dea'] = dea
    result['lon'] = lon_hist
    result['lonma'] = lonma
    result['lon_len'] = lon_hist.abs()
    return result
