"""
股票回测框架使用示例

演示如何使用股票回测项目进行：
1. 获取股票数据
2. 计算技术指标
3. 运行策略回测
4. 查看回测结果
"""

import pandas as pd
import sys
sys.path.insert(0, '/Users/aqichita/projects/stock-backtest')

# ==================== 1. 获取股票数据 ====================
from src.data.manager import DataManager

print("=" * 50)
print("步骤1: 获取股票数据")
print("=" * 50)

dm = DataManager()

# 获取平安银行(000001.SZ)的历史数据
df = dm.get_daily_data(
    symbol="000001.SZ",
    start="2023-01-01",
    end="2024-12-31"
)

print(f"获取到 {len(df)} 条数据")
print(f"数据列: {list(df.columns)}")
print(f"\n前5行数据:")
print(df.head())
print()

# ==================== 2. 计算技术指标 ====================
from src.indicators import (
    sma, ema, macd, rsi, bollinger_bands, kdj, atr,
    max_force_resonance, xpct_only
)

print("=" * 50)
print("步骤2: 计算技术指标")
print("=" * 50)

# 计算MA均线
df['ma5'] = sma(df, period=5)
df['ma20'] = sma(df, period=20)
df['ma60'] = sma(df, period=60)

# 计算MACD
df['macd'], df['macd_signal'], df['macd_hist'] = macd(df)

# 计算RSI
df['rsi'] = rsi(df, period=14)

# 计算布林带
df['bb_upper'], df['bb_middle'], df['bb_lower'] = bollinger_bands(df)

# 计算KDJ
df['k'], df['d'], df['j'] = kdj(df)

# 计算ATR
df['atr'] = atr(df)

# 计算主力共振指标
mfr = max_force_resonance(df, n=12, m=240)
df['xpct'] = mfr['xpct']
df['main_net_inflow'] = mfr['main_net_inflow']
df['strong_buy'] = mfr['strong_buy']
df['xg100_s'] = mfr['xg100_s']

print("技术指标计算完成!")
print(f"\n最新数据指标值:")
latest = df.iloc[-1]
print(f"  收盘价: {latest['close']:.2f}")
print(f"  MA5: {latest['ma5']:.2f}")
print(f"  MA20: {latest['ma20']:.2f}")
print(f"  RSI: {latest['rsi']:.2f}")
print(f"  MACD: {latest['macd']:.4f}")
print(f"  XPCT: {latest['xpct']:.2f}")
print(f"  主力净流入: {latest['main_net_inflow']:.2f}")
print()

# ==================== 3. 运行策略回测 ====================
from src.strategies import (
    BuyAndHoldStrategy,
    MovingAverageCrossStrategy,
    RSIStrategy,
    MACDStrategy,
    BollingerStrategy,
    KDJStrategy,
    MainForceResonanceStrategy
)
from src.backtest.engine import BacktestEngine

print("=" * 50)
print("步骤3: 运行策略回测")
print("=" * 50)

# 初始化回测引擎
engine = BacktestEngine(initial_cash=100000)

# 定义要测试的策略
strategies = {
    '买入持有': BuyAndHoldStrategy(),
    '均线交叉': MovingAverageCrossStrategy(short_window=5, long_window=20),
    'RSI策略': RSIStrategy(rsi_period=14, oversold=30, overbought=70),
    'MACD策略': MACDStrategy(),
    '布林带策略': BollingerStrategy(),
    'KDJ策略': KDJStrategy(),
    '主力共振策略': MainForceResonanceStrategy(n=12, m=240),
}

results = {}

for name, strategy in strategies.items():
    print(f"\n正在回测: {name}...")
    result = engine.run(strategy=strategy, data=df)
    results[name] = result
    
    print(f"  初始资金: ¥{result['initial_cash']:,.2f}")
    print(f"  最终资产: ¥{result['final_value']:,.2f}")
    print(f"  总收益率: {result['total_return']*100:.2f}%")
    print(f"  交易次数: {result['trade_count']}")

# ==================== 4. 对比策略表现 ====================
print("\n" + "=" * 50)
print("步骤4: 策略对比")
print("=" * 50)

print("\n策略表现排名:")
sorted_results = sorted(results.items(), key=lambda x: x[1]['total_return'], reverse=True)

for i, (name, result) in enumerate(sorted_results, 1):
    print(f"{i}. {name:12s} - 收益率: {result['total_return']*100:+.2f}%  (交易{result['trade_count']}次)")

# ==================== 5. 查看详细信号 ====================
print("\n" + "=" * 50)
print("步骤5: 查看最近的交易信号")
print("=" * 50)

# 使用主力共振策略生成详细信号
strategy = MainForceResonanceStrategy(n=12, m=240)
signals = strategy.generate_signals(df)

# 合并到df
df_with_signals = df.copy()
df_with_signals['signal'] = signals['signal']
df_with_signals['signal_type'] = signals['signal_type']

# 找出最近的交易信号
recent_signals = df_with_signals[df_with_signals['signal'] != 0].tail(5)

print("\n最近5个交易信号:")
for idx, row in recent_signals.iterrows():
    date = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
    action = "买入" if row['signal'] == 1 else "卖出"
    print(f"  {date} - {action} - {row['signal_type']} - 收盘价: ¥{row['close']:.2f}")

print("\n" + "=" * 50)
print("示例运行完成!")
print("=" * 50)
