"""
回测结果可视化模块

支持 matplotlib 和 plotly 两种后端
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List


def plot_backtest_result(
    result: Dict,
    show_trades: bool = True,
    backend: str = "matplotlib",
    figsize: tuple = (14, 10)
):
    """
    绘制回测结果图表
    
    Args:
        result: 回测引擎返回的结果字典
        show_trades: 是否标注交易点
        backend: "matplotlib" 或 "plotly"
        figsize: 图表尺寸
    """
    if backend == "matplotlib":
        return _plot_with_matplotlib(result, show_trades, figsize)
    elif backend == "plotly":
        return _plot_with_plotly(result, show_trades)
    else:
        raise ValueError(f"不支持的可视化后端: {backend}")


def _plot_with_matplotlib(result: Dict, show_trades: bool, figsize: tuple):
    """使用 matplotlib 绘制回测结果"""
    import matplotlib.pyplot as plt
    
    daily_values = result.get('daily_values')
    trades = result.get('trades')
    
    if daily_values is None or daily_values.empty:
        print("没有可绘制的数据")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, gridspec_kw={'height_ratios': [2, 1, 1]})
    fig.suptitle(f"回测结果 | 总收益: {result.get('total_return_pct', 'N/A')} | 夏普: {result.get('sharpe_ratio', 0):.2f}", fontsize=14)
    
    # 1. 资产净值曲线
    ax1 = axes[0]
    ax1.plot(daily_values['date'], daily_values['total_value'], label='总资产', color='#2196F3', linewidth=1.5)
    ax1.fill_between(daily_values['date'], daily_values['total_value'], alpha=0.1, color='#2196F3')
    ax1.axhline(y=result.get('initial_cash', 100000), color='gray', linestyle='--', alpha=0.5, label='初始资金')
    
    if show_trades and trades is not None and not trades.empty:
        buy_trades = trades[trades['direction'] == 'buy']
        sell_trades = trades[trades['direction'] == 'sell']
        
        if not buy_trades.empty:
            ax1.scatter(buy_trades['date'], [daily_values.set_index('date').loc[d, 'total_value'] 
                        if d in daily_values['date'].values else np.nan 
                        for d in buy_trades['date']], 
                       color='green', marker='^', s=80, label='买入', zorder=5)
        
        if not sell_trades.empty:
            ax1.scatter(sell_trades['date'], [daily_values.set_index('date').loc[d, 'total_value'] 
                        if d in daily_values['date'].values else np.nan 
                        for d in sell_trades['date']], 
                       color='red', marker='v', s=80, label='卖出', zorder=5)
    
    ax1.set_ylabel('资产价值')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. 回撤曲线
    ax2 = axes[1]
    daily_values['cummax'] = daily_values['total_value'].cummax()
    daily_values['drawdown'] = (daily_values['total_value'] - daily_values['cummax']) / daily_values['cummax']
    
    ax2.fill_between(daily_values['date'], daily_values['drawdown'], 0, color='#F44336', alpha=0.3)
    ax2.plot(daily_values['date'], daily_values['drawdown'], color='#F44336', linewidth=1)
    ax2.set_ylabel('回撤')
    ax2.set_ylim(daily_values['drawdown'].min() * 1.2, 0.05)
    ax2.grid(True, alpha=0.3)
    
    # 3. 现金与持仓价值
    ax3 = axes[2]
    ax3.plot(daily_values['date'], daily_values['cash'], label='现金', color='#4CAF50', linewidth=1)
    ax3.plot(daily_values['date'], daily_values['positions_value'], label='持仓价值', color='#FF9800', linewidth=1)
    ax3.set_ylabel('金额')
    ax3.set_xlabel('日期')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    
    return fig


def _plot_with_plotly(result: Dict, show_trades: bool):
    """使用 plotly 绘制交互式回测结果"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("请先安装 plotly: pip install plotly")
    
    daily_values = result.get('daily_values')
    trades = result.get('trades')
    
    if daily_values is None or daily_values.empty:
        print("没有可绘制的数据")
        return
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('资产净值', '回撤', '现金与持仓'),
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # 1. 资产净值
    fig.add_trace(
        go.Scatter(x=daily_values['date'], y=daily_values['total_value'],
                   name='总资产', line=dict(color='#2196F3', width=1.5),
                   fill='tozeroy', fillcolor='rgba(33, 150, 243, 0.1)'),
        row=1, col=1
    )
    
    if show_trades and trades is not None and not trades.empty:
        buy_trades = trades[trades['direction'] == 'buy']
        sell_trades = trades[trades['direction'] == 'sell']
        
        if not buy_trades.empty:
            buy_values = [daily_values.set_index('date').loc[d, 'total_value'] 
                         if d in daily_values['date'].values else None 
                         for d in buy_trades['date']]
            fig.add_trace(
                go.Scatter(x=buy_trades['date'], y=buy_values,
                           mode='markers', name='买入',
                           marker=dict(color='green', symbol='triangle-up', size=10)),
                row=1, col=1
            )
        
        if not sell_trades.empty:
            sell_values = [daily_values.set_index('date').loc[d, 'total_value'] 
                          if d in daily_values['date'].values else None 
                          for d in sell_trades['date']]
            fig.add_trace(
                go.Scatter(x=sell_trades['date'], y=sell_values,
                           mode='markers', name='卖出',
                           marker=dict(color='red', symbol='triangle-down', size=10)),
                row=1, col=1
            )
    
    # 2. 回撤
    daily_values['cummax'] = daily_values['total_value'].cummax()
    daily_values['drawdown'] = (daily_values['total_value'] - daily_values['cummax']) / daily_values['cummax']
    
    fig.add_trace(
        go.Scatter(x=daily_values['date'], y=daily_values['drawdown'],
                   name='回撤', line=dict(color='#F44336', width=1),
                   fill='tozeroy', fillcolor='rgba(244, 67, 54, 0.3)'),
        row=2, col=1
    )
    
    # 3. 现金与持仓
    fig.add_trace(
        go.Scatter(x=daily_values['date'], y=daily_values['cash'],
                   name='现金', line=dict(color='#4CAF50', width=1)),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=daily_values['date'], y=daily_values['positions_value'],
                   name='持仓价值', line=dict(color='#FF9800', width=1)),
        row=3, col=1
    )
    
    fig.update_layout(
        title=f"回测结果 | 总收益: {result.get('total_return_pct', 'N/A')} | 夏普: {result.get('sharpe_ratio', 0):.2f}",
        hovermode='x unified',
        showlegend=True,
        height=700
    )
    
    fig.show()
    return fig


def plot_price_with_signals(data: pd.DataFrame, trades: Optional[pd.DataFrame] = None):
    """
    绘制价格走势与交易信号
    
    Args:
        data: 日线数据 DataFrame
        trades: 交易记录 DataFrame
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(data['date'], data['close'], label='收盘价', color='#2196F3', linewidth=1.5)
    
    if trades is not None and not trades.empty:
        buy_trades = trades[trades['direction'] == 'buy']
        sell_trades = trades[trades['direction'] == 'sell']
        
        if not buy_trades.empty:
            ax.scatter(buy_trades['date'], buy_trades['price'], 
                      color='green', marker='^', s=100, label='买入', zorder=5)
        
        if not sell_trades.empty:
            ax.scatter(sell_trades['date'], sell_trades['price'], 
                      color='red', marker='v', s=100, label='卖出', zorder=5)
    
    ax.set_xlabel('日期')
    ax.set_ylabel('价格')
    ax.set_title('价格走势与交易信号')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_strategy_comparison(results: List[Dict], labels: List[str]):
    """
    对比多个策略的回测结果
    
    Args:
        results: 多个回测结果字典
        labels: 策略标签
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
    
    # 1. 净值对比
    ax1 = axes[0]
    for i, (result, label) in enumerate(zip(results, labels)):
        daily_values = result.get('daily_values')
        if daily_values is not None and not daily_values.empty:
            color = colors[i % len(colors)]
            ax1.plot(daily_values['date'], daily_values['total_value'], 
                    label=label, color=color, linewidth=1.5)
    
    ax1.set_ylabel('资产价值')
    ax1.set_title('策略净值对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 指标对比柱状图
    ax2 = axes[1]
    metrics = ['total_return', 'max_drawdown', 'sharpe_ratio']
    metric_labels = ['总收益率', '最大回撤', '夏普比率']
    x = np.arange(len(labels))
    width = 0.25
    
    for j, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        values = [result.get(metric, 0) * (100 if metric == 'total_return' else 1) for result in results]
        ax2.bar(x + j * width, values, width, label=metric_label, color=colors[j])
    
    ax2.set_ylabel('数值')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    return fig
