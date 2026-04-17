# 股票回测项目 - 使用指南

## 快速开始

### 第一步：下载数据到本地（只需执行一次）

```bash
cd /Users/aqichita/projects/stock-backtest

# 下载默认股票列表（约20只热门股票）
python download_data.py --start 2020-01-01 --end 2024-12-31

# 下载指定股票
python download_data.py --symbols 000001.SZ,600519.SH,00700.HK

# 下载沪深300成分股
python download_data.py --index-components hs300

# 下载全部A股（约5000+只，耗时较长）
python download_data.py --index-components a --limit 100
```

数据将保存在 `data/parquet/` 目录下，格式为 `.parquet` 文件。

### 第二步：快速回测（使用本地数据，速度极快）

```bash
# 运行快速回测示例
python quick_backtest.py
```

## 代码中使用

### 1. 下载数据脚本

```python
from src.data.manager import DataManager
from src.data.storage import DataStorage

# 下载单只股票
dm = DataManager()
storage = DataStorage(backend="parquet")

df = dm.get_daily_data("000001.SZ", start="2020-01-01", end="2024-12-31")
storage.save_to_parquet(df, "000001.SZ")
```

### 2. 从本地加载数据进行回测

```python
from src.data.storage import DataStorage
from src.strategies import MainForceResonanceStrategy
from src.backtest.engine import BacktestEngine

# 从本地加载（无需网络，秒开）
storage = DataStorage(backend="parquet")
df = storage.load_from_parquet("000001.SZ", start="2023-01-01")

# 运行回测
engine = BacktestEngine(initial_cash=100000)
strategy = MainForceResonanceStrategy(n=12, m=240)
result = engine.run(strategy=strategy, data=df)

print(f"收益率: {result['total_return']*100:.2f}%")
```

## 命令行工具

### download_data.py - 数据下载

```bash
# 基础用法
python download_data.py

# 指定时间范围
python download_data.py --start 2022-01-01 --end 2024-12-31

# 指定股票
python download_data.py --symbols 000001.SZ,600519.SH

# 下载指数成分股
python download_data.py --index-components hs300    # 沪深300
python download_data.py --index-components zz500    # 中证500
python download_data.py --index-components a        # 全部A股

# 限制数量（测试用）
python download_data.py --index-components a --limit 50

# 强制更新（重新下载已存在的）
python download_data.py --symbols 000001.SZ --force

# 调整并发数（默认4）
python download_data.py --workers 8
```

### quick_backtest.py - 快速回测

直接运行查看示例：
```bash
python quick_backtest.py
```

或作为模块导入使用：
```python
from quick_backtest import quick_backtest, batch_backtest

# 单股票多策略对比
quick_backtest("000001.SZ", start_date="2023-01-01")

# 多股票批量回测
batch_backtest(
    symbols=["000001.SZ", "600519.SH", "300750.SZ"],
    strategy_name="主力共振"
)
```

## 数据存储说明

- **位置**: `data/parquet/`
- **格式**: Parquet（压缩率高，读取速度快）
- **命名**: `000001_SZ.parquet`（点号替换为下划线）
- **更新**: 重复下载会自动合并，去重保留最新数据

## 性能对比

| 操作 | 网络实时获取 | 本地Parquet |
|------|-------------|-------------|
| 加载单股票 | 3-10秒 | <0.1秒 |
| 回测100只股票 | 5-15分钟 | <10秒 |

**建议**: 先批量下载数据到本地，之后所有回测都从本地读取。

## 可用策略列表

| 策略 | 说明 |
|------|------|
| BuyAndHoldStrategy | 买入持有（基准） |
| MovingAverageCrossStrategy | 均线交叉 |
| RSIStrategy | RSI超买超卖 |
| MACDStrategy | MACD金叉死叉 |
| BollingerStrategy | 布林带突破 |
| KDJStrategy | KDJ指标 |
| ATRStrategy | ATR波动率 |
| MainForceResonanceStrategy | 主力共振（自定义） |

## 常见问题

**Q: 数据多久更新一次？**
A: 建议每天收盘后运行一次下载脚本更新数据。

**Q: 可以下载港股/美股吗？**
A: 港股支持（如00700.HK），美股需要接入其他数据源。

**Q: 数据文件有多大？**
A: 单只股票约100KB-1MB（取决于时间跨度），1000只股票约500MB。

**Q: 如何清空本地数据？**
A: 删除 `data/parquet/` 目录下的 `.parquet` 文件即可。
