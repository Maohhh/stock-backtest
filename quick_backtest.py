"""
使用本地数据进行快速回测

这个脚本演示如何：
1. 从本地加载已下载的股票数据
2. 运行多个策略回测
3. 对比策略表现
"""

import sys
sys.path.insert(0, '/Users/aqichita/projects/stock-backtest')

import pandas as pd
from datetime import datetime

from src.data.storage import DataStorage
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


def quick_backtest(
    symbol: str,
    start_date: str = None,
    end_date: str = None,
    initial_cash: float = 100000
):
    """
    快速回测单个股票
    
    从本地加载数据，运行所有策略，返回对比结果
    """
    storage = DataStorage()
    
    # 从本地加载数据（无需网络请求，速度极快）
    print(f"正在从本地加载 {symbol} 数据...")
    df = storage.load_from_parquet(symbol, start=start_date, end=end_date)
    
    if df is None or df.empty:
        print(f"❌ 本地没有找到 {symbol} 的数据，请先运行 download_data.py 下载")
        return None
    
    print(f"✅ 成功加载 {len(df)} 条数据 ({df['date'].min()} ~ {df['date'].max()})")
    
    # 定义策略
    strategies = {
        '买入持有': BuyAndHoldStrategy(),
        '均线交叉(5/20)': MovingAverageCrossStrategy(short_window=5, long_window=20),
        '均线交叉(10/60)': MovingAverageCrossStrategy(short_window=10, long_window=60),
        'RSI(14)': RSIStrategy(rsi_period=14, oversold=30, overbought=70),
        'MACD': MACDStrategy(),
        '布林带': BollingerStrategy(),
        'KDJ': KDJStrategy(),
        '主力共振': MainForceResonanceStrategy(n=12, m=240),
    }
    
    # 运行回测
    print(f"\n开始回测（初始资金: ¥{initial_cash:,.0f}）...")
    print("-" * 80)
    
    results = []
    engine = BacktestEngine(initial_cash=initial_cash)
    
    for name, strategy in strategies.items():
        result = engine.run(strategy=strategy, data=df)
        results.append({
            'strategy': name,
            'final_value': result['final_value'],
            'total_return': result['total_return'],
            'trade_count': result['trade_count'],
            'win_rate': result.get('win_rate', 0),
        })
    
    # 排序并显示结果
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('total_return', ascending=False)
    
    print(f"\n{'策略':<20} {'最终资产':>12} {'收益率':>10} {'交易次数':>8} {'胜率':>8}")
    print("-" * 80)
    
    for _, row in results_df.iterrows():
        print(f"{row['strategy']:<20} "
              f"¥{row['final_value']:>10,.0f} "
              f"{row['total_return']*100:>+9.2f}% "
              f"{row['trade_count']:>8.0f} "
              f"{row['win_rate']*100:>7.1f}%")
    
    print("-" * 80)
    
    # 找出最佳策略
    best = results_df.iloc[0]
    print(f"\n🏆 最佳策略: {best['strategy']}")
    print(f"   收益率: {best['total_return']*100:+.2f}%")
    print(f"   最终资产: ¥{best['final_value']:,.0f}")
    
    return results_df


def batch_backtest(
    symbols: list,
    start_date: str = None,
    end_date: str = None,
    strategy_name: str = '主力共振'
):
    """
    批量回测多只股票
    
    使用指定策略回测多只股票，对比表现
    """
    storage = DataStorage()
    
    # 选择策略
    strategies = {
        '买入持有': BuyAndHoldStrategy(),
        '均线交叉': MovingAverageCrossStrategy(short_window=5, long_window=20),
        'RSI': RSIStrategy(rsi_period=14, oversold=30, overbought=70),
        'MACD': MACDStrategy(),
        '布林带': BollingerStrategy(),
        'KDJ': KDJStrategy(),
        '主力共振': MainForceResonanceStrategy(n=12, m=240),
    }
    
    strategy = strategies.get(strategy_name)
    if not strategy:
        print(f"❌ 未知策略: {strategy_name}")
        return None
    
    print(f"批量回测 {len(symbols)} 只股票，使用策略: {strategy_name}")
    print("-" * 80)
    
    results = []
    engine = BacktestEngine(initial_cash=100000)
    
    for symbol in symbols:
        df = storage.load_from_parquet(symbol, start=start_date, end=end_date)
        if df is None or df.empty:
            print(f"⏭️  {symbol}: 无本地数据")
            continue
        
        result = engine.run(strategy=strategy, data=df)
        results.append({
            'symbol': symbol,
            'days': len(df),
            'final_value': result['final_value'],
            'total_return': result['total_return'],
            'trade_count': result['trade_count'],
        })
        
        print(f"✅ {symbol}: {result['total_return']*100:+.2f}% ({len(df)}天)")
    
    if not results:
        print("❌ 没有成功回测任何股票")
        return None
    
    # 汇总
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('total_return', ascending=False)
    
    print("-" * 80)
    print(f"\n汇总统计 ({len(results)} 只股票):")
    print(f"  平均收益率: {results_df['total_return'].mean()*100:+.2f}%")
    print(f"  最高收益率: {results_df['total_return'].max()*100:+.2f}% ({results_df.iloc[0]['symbol']})")
    print(f"  最低收益率: {results_df['total_return'].min()*100:+.2f}% ({results_df.iloc[-1]['symbol']})")
    print(f"  胜率: {(results_df['total_return'] > 0).mean()*100:.1f}%")
    
    return results_df


def main():
    """主函数 - 演示使用"""
    
    # 示例1: 单股票多策略回测
    print("=" * 80)
    print("示例1: 单股票多策略回测")
    print("=" * 80)
    quick_backtest(
        symbol="000001.SZ",  # 平安银行
        start_date="2023-01-01",
        end_date="2024-12-31"
    )
    
    # 示例2: 多股票批量回测
    print("\n" + "=" * 80)
    print("示例2: 多股票批量回测（使用主力共振策略）")
    print("=" * 80)
    
    symbols = [
        "000001.SZ",  # 平安银行
        "600519.SH",  # 贵州茅台
        "300750.SZ",  # 宁德时代
        "002594.SZ",  # 比亚迪
    ]
    
    batch_backtest(
        symbols=symbols,
        start_date="2023-01-01",
        end_date="2024-12-31",
        strategy_name="主力共振"
    )


if __name__ == '__main__':
    main()
