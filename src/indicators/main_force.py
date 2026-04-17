"""
主力共振指标 (Main Force Resonance Indicator)

这是一个复合技术指标，结合了：
1. XPCT - 基于改良RSI的百分比位置指标
2. 主力进场/洗盘/出货信号
3. 主力净流入计算
4. 买卖共振判断

用于识别主力资金动向和买卖时机。
"""

import pandas as pd
import numpy as np


def max_force_resonance(df: pd.DataFrame, 
                        n: int = 12, 
                        m: int = 240,
                        bp_buy: float = 0,
                        sp_sell: float = 95) -> pd.DataFrame:
    """
    计算主力共振指标
    
    参数:
        df: 包含股票数据的DataFrame，需要列：open, high, low, close, volume
        n: XPCT计算周期，默认12
        m: XPCT高低点计算周期，默认240
        bp_buy: 买入阈值，默认0
        sp_sell: 卖出阈值，默认95
    
    返回:
        DataFrame包含以下列：
        - xpct: 百分比位置指标 (0-100)
        - main_force_entry: 主力进场信号 (True/False)
        - wash: 洗盘信号 (True/False)
        - ship: 出货信号 (True/False)
        - main_net_inflow: 主力净流入
        - weak_buy: 弱买点信号 (True/False)
        - strong_buy: 强买点信号 (True/False)
        - sell_resonance: 卖共振信号 (True/False)
        - xg100_s: 卖出信号 (True/False)
    
    示例:
        >>> result = max_force_resonance(df)
        >>> buy_signals = result[result['strong_buy'] == True]
        >>> sell_signals = result[result['xg100_s'] == True]
    """
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必需列: {col}")
    
    result = pd.DataFrame(index=df.index)
    
    # ========== XPCT 计算 ==========
    # A = SMA(MAX(C-REF(C,1),0),N,1)*4 - SMA(ABS(C-REF(C,1)),N,1)
    c = df['close']
    price_change = c.diff()
    max_gain = np.maximum(price_change, 0)
    abs_change = price_change.abs()
    
    # SMA (简单移动平均)
    sma_max_gain = max_gain.rolling(window=n, min_periods=1).mean()
    sma_abs_change = abs_change.rolling(window=n, min_periods=1).mean()
    a = sma_max_gain * 4 - sma_abs_change
    
    # XMIN = LLV(A,M), XMAX = HHV(A,M)
    x_min = a.rolling(window=m, min_periods=1).min()
    x_max = a.rolling(window=m, min_periods=1).max()
    
    # XRNG = XMAX - XMIN
    x_rng = x_max - x_min
    
    # XPCT = IF(XRNG=0, 50, (A-XMIN)/XRNG*100)
    result['xpct'] = np.where(x_rng == 0, 50, (a - x_min) / x_rng * 100)
    
    # ========== 主力进场/洗盘计算 ==========
    # VAR1 = (LOW+OPEN+CLOSE+HIGH)/4
    var1 = (df['low'] + df['open'] + df['close'] + df['high']) / 4
    var2 = var1.shift(1)
    
    # VAR3 = SMA(ABS(LOW-VAR2),13,1) / SMA(MAX(LOW-VAR2,0),10,1)
    low_diff = (df['low'] - var2).abs()
    sma_low_diff = low_diff.rolling(window=13, min_periods=1).mean()
    max_low_diff = np.maximum(df['low'] - var2, 0)
    sma_max_low_diff = max_low_diff.rolling(window=10, min_periods=1).mean()
    var3 = sma_low_diff / sma_max_low_diff.replace(0, np.nan)
    
    # VAR4 = EMA(VAR3,10)
    var4 = var3.ewm(span=10, min_periods=1, adjust=False).mean()
    
    # VAR5 = LLV(LOW,33)
    var5 = df['low'].rolling(window=33, min_periods=1).min()
    
    # VAR6 = EMA(IF(LOW<=VAR5,VAR4,0),3)
    var6_cond = var4.where(df['low'] <= var5, 0)
    var6 = var6_cond.ewm(span=3, min_periods=1, adjust=False).mean()
    
    # 主力进场 = VAR6 > REF(VAR6,1)
    result['main_force_entry'] = var6 > var6.shift(1)
    
    # 洗盘 = VAR6 < REF(VAR6,1)
    result['wash'] = var6 < var6.shift(1)
    
    # ========== 出货计算 ==========
    # VAR21 = SMA(ABS(HIGH-VAR2),13,1) / SMA(MIN(HIGH-VAR2,0),10,1)
    high_diff = (df['high'] - var2).abs()
    sma_high_diff = high_diff.rolling(window=13, min_periods=1).mean()
    min_high_diff = np.minimum(df['high'] - var2, 0)
    sma_min_high_diff = min_high_diff.abs().rolling(window=10, min_periods=1).mean()
    var21 = sma_high_diff / sma_min_high_diff.replace(0, np.nan)
    
    # VAR31 = EMA(VAR21,10)
    var31 = var21.ewm(span=10, min_periods=1, adjust=False).mean()
    
    # VAR41 = HHV(HIGH,33)
    var41 = df['high'].rolling(window=33, min_periods=1).max()
    
    # VAR51 = EMA(IF(HIGH>=VAR41,VAR31,0),3)
    var51_cond = var31.where(df['high'] >= var41, 0)
    var51 = var51_cond.ewm(span=3, min_periods=1, adjust=False).mean()
    
    # 出货 = VAR51 > REF(VAR51,1)
    result['ship'] = var51 > var51.shift(1)
    
    # ========== 主力净流入计算 ==========
    # JJ = (HIGH+LOW+CLOSE)/3
    jj = (df['high'] + df['low'] + df['close']) / 3
    
    # QJ0 = VOL/IF(HIGH=LOW,4,HIGH-LOW)
    hl_diff = np.where(df['high'] == df['low'], 4, df['high'] - df['low'])
    qj0 = df['volume'] / hl_diff
    
    # 主买 = QJ0 * (JJ - MIN(CLOSE,OPEN))
    main_buy = qj0 * (jj - np.minimum(df['close'], df['open']))
    
    # 主卖 = QJ0 * (MIN(OPEN,CLOSE) - LOW)
    main_sell = qj0 * (np.minimum(df['open'], df['close']) - df['low'])
    
    # TMP = 主买 - 主卖
    tmp = main_buy - main_sell
    
    # AA = VOL / IF(((HIGH-LOW)*2-ABS(CLOSE-OPEN))=0, 1, ((HIGH-LOW)*2-ABS(CLOSE-OPEN)))
    hl_range = (df['high'] - df['low']) * 2 - (df['close'] - df['open']).abs()
    aa = df['volume'] / np.where(hl_range == 0, 1, hl_range)
    
    # 买量计算
    buy_vol = np.where(
        df['close'] > df['open'],
        aa * (df['high'] - df['low']),
        np.where(
            df['close'] < df['open'],
            aa * ((df['high'] - df['open']) + (df['close'] - df['low'])),
            df['volume'] / 2
        )
    )
    
    # 卖量计算
    sell_vol = np.where(
        df['close'] > df['open'],
        -aa * ((df['high'] - df['close']) + (df['open'] - df['low'])),
        np.where(
            df['close'] < df['open'],
            -aa * (df['high'] - df['low']),
            -df['volume'] / 2
        )
    )
    
    # 主力净流入 = MA(买量+卖量/2, 3)
    result['main_net_inflow'] = pd.Series(buy_vol + sell_vol / 2, index=df.index).rolling(window=3, min_periods=1).mean()
    
    # ========== 买卖信号计算 ==========
    # 弱买点 = XPCT <= BP_BUY
    result['weak_buy'] = result['xpct'] <= bp_buy
    
    # 强买点 = XPCT <= BP_BUY AND 主力净流入 > 0
    result['strong_buy'] = (result['xpct'] <= bp_buy) & (result['main_net_inflow'] > 0)
    
    # 卖共振 = XPCT >= SP_SELL AND 出货
    result['sell_resonance'] = (result['xpct'] >= sp_sell) & result['ship']
    
    # XG100_S = 卖共振 AND (TMP < 0 OR 主力净流入 < 0)
    result['xg100_s'] = result['sell_resonance'] & ((tmp < 0) | (result['main_net_inflow'] < 0))
    
    return result


def xpct_only(df: pd.DataFrame, n: int = 12, m: int = 240) -> pd.Series:
    """
    仅计算XPCT指标（简化版）
    
    XPCT是一个0-100的百分比位置指标，类似于改良版的RSI。
    - 0表示极低位（超卖）
    - 50表示中间位置
    - 100表示极高位（超买）
    
    参数:
        df: 包含股票数据的DataFrame
        n: 计算周期，默认12
        m: 高低点计算周期，默认240
    
    返回:
        XPCT值的Series (0-100)
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame需要包含'close'列")
    
    c = df['close']
    price_change = c.diff()
    max_gain = np.maximum(price_change, 0)
    abs_change = price_change.abs()
    
    sma_max_gain = max_gain.rolling(window=n, min_periods=1).mean()
    sma_abs_change = abs_change.rolling(window=n, min_periods=1).mean()
    a = sma_max_gain * 4 - sma_abs_change
    
    x_min = a.rolling(window=m, min_periods=1).min()
    x_max = a.rolling(window=m, min_periods=1).max()
    x_rng = x_max - x_min
    
    xpct = np.where(x_rng == 0, 50, (a - x_min) / x_rng * 100)
    return pd.Series(xpct, index=df.index)
