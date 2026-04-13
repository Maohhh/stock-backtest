# Stock Backtest

A股港股历史数据获取与回测框架

## 项目结构

```
stock-backtest/
├── src/
│   ├── data/           # 数据获取模块
│   │   ├── __init__.py
│   │   ├── sina.py     # Sina 数据源
│   │   ├── akshare.py  # AkShare 数据源
│   │   └── manager.py  # 数据管理器（自动 fallback）
│   ├── backtest/       # 回测框架
│   │   ├── __init__.py
│   │   ├── engine.py   # 回测引擎
│   │   ├── portfolio.py # 投资组合
│   │   └── metrics.py  # 绩效指标
│   ├── strategies/     # 策略模块
│   │   ├── __init__.py
│   │   └── base.py     # 策略基类
│   └── utils/          # 工具函数
│       ├── __init__.py
│       └── helpers.py
├── tests/              # 测试
├── notebooks/          # Jupyter 笔记本
├── requirements.txt    # 依赖
└── README.md           # 本文件
```

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

```python
from src.data.manager import DataManager
from src.backtest.engine import BacktestEngine
from src.strategies.base import BaseStrategy

# 获取数据
dm = DataManager()
df = dm.get_daily_data("000001.SZ", start="2020-01-01", end="2024-01-01")

# 运行回测
engine = BacktestEngine(initial_cash=100000)
engine.run(strategy=MyStrategy(), data=df)
```

## 数据源

- **Sina**: 实时数据，免费但有限制
- **AkShare**: 开源财经数据接口，作为 fallback

## 许可证

MIT