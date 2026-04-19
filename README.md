# Stock Backtest

A 股历史数据获取、技术指标计算、策略回测和结果可视化框架。

## 功能概览

- 数据获取：支持 Baostock、AkShare、Sina，`auto` 模式优先使用 Baostock，失败后自动 fallback。
- 本地存储：支持 Parquet 和 SQLite，便于批量下载后离线回测。
- 技术指标：已实现 MA、MACD、RSI、布林带、KDJ、ATR、主力共振/XPCT 等指标。
- 策略模块：内置买入持有、均线交叉、MACD、RSI、布林带、KDJ、ATR、主力共振策略。
- 回测引擎：支持单标的日线回测，输出收益率、最大回撤、夏普比率、交易记录和每日净值。
- 可视化：支持 matplotlib 和 plotly 绘制净值、回撤、仓位资金、交易点和策略对比。

## 项目结构

```text
stock-backtest/
├── src/
│   ├── data/
│   │   ├── manager.py      # Baostock / AkShare / Sina 数据源和 DataManager
│   │   └── storage.py      # SQLite / Parquet 本地存储
│   ├── indicators/
│   │   ├── ma.py           # SMA / EMA
│   │   ├── macd.py         # MACD
│   │   ├── rsi.py          # RSI
│   │   ├── bollinger.py    # 布林带
│   │   ├── kdj.py          # KDJ
│   │   ├── atr.py          # ATR
│   │   └── main_force.py   # 主力共振 / XPCT
│   ├── strategies/
│   │   ├── base.py                 # BaseStrategy / 买入持有 / 均线交叉
│   │   ├── macd_strategy.py        # MACD 金叉死叉策略
│   │   ├── rsi_strategy.py         # RSI 阈值穿越策略
│   │   ├── bollinger_strategy.py   # 布林带策略
│   │   ├── kdj_strategy.py         # KDJ 策略
│   │   ├── atr_strategy.py         # ATR 波动率突破策略
│   │   └── main_force_strategy.py  # 主力共振策略
│   ├── backtest/
│   │   └── engine.py       # BacktestEngine
│   └── visualization/
│       └── charts.py       # 回测图表、交易信号、策略对比
├── download_data.py        # 批量下载日线数据
├── quick_backtest.py       # 快速回测示例
├── example_usage.py        # 综合示例
├── USAGE.md                # 详细使用指南
├── tests/
└── requirements.txt
```

## 安装

```bash
cd /Users/aqichita/projects/stock-backtest
pip install -r requirements.txt
```

如果使用 Parquet 存储，需要额外安装 Parquet 引擎：

```bash
pip install pyarrow
```

## 数据获取

`DataManager.get_daily_data()` 返回标准日线 DataFrame，列为 `date/open/high/low/close/volume`。

```python
from src.data.manager import DataManager

dm = DataManager()
df = dm.get_daily_data(
    symbol="000001.SZ",
    start="2023-01-01",
    end="2024-12-31",
    source="auto",  # auto / baostock / akshare / sina
)

df.attrs["symbol"] = "000001.SZ"
print(df.head())
```

批量下载到本地：

```bash
# 下载默认股票列表
python download_data.py --start 2020-01-01 --end 2024-12-31

# 下载指定股票
python download_data.py --symbols 000001.SZ,600519.SH,300750.SZ --source auto

# 下载沪深300、中证500或全部A股候选列表
python download_data.py --index-components hs300 --limit 50
python download_data.py --index-components zz500 --random 20
python download_data.py --index-components a --limit 100

# 指定存储后端
python download_data.py --symbols 000001.SZ --storage-backend sqlite
```

## 技术指标

所有指标可从 `src.indicators` 统一导入。

| 指标 | 函数 | 输入要求 | 返回 |
| --- | --- | --- | --- |
| 简单移动平均 | `sma(df, period=20, column="close")` | 指定价格列 | `pd.Series` |
| 指数移动平均 | `ema(df, period=20, column="close")` | 指定价格列 | `pd.Series` |
| MACD | `macd(df, fast=12, slow=26, signal=9, column="close")` | 指定价格列 | 含 `DIF/DEA/MACD` 的 DataFrame |
| RSI | `rsi(df, period=14, column="close")` | 指定价格列 | `pd.Series`，范围 0-100 |
| 布林带 | `bollinger_bands(df, period=20, std_dev=2.0, column="close")` | 指定价格列 | 含 `middle/upper/lower` 的 DataFrame |
| KDJ | `kdj(df, n=9, m1=3, m2=3)` | `high/low/close` | 含 `K/D/J` 的 DataFrame |
| ATR | `atr(df, period=14)` | `high/low/close` | `pd.Series` |
| 主力共振 | `max_force_resonance(df, n=12, m=240, bp_buy=0, sp_sell=95)` | `open/high/low/close/volume` | 含 XPCT、主力净流入、买卖信号的 DataFrame |
| XPCT | `xpct_only(df, n=12, m=240)` | `close` | `pd.Series`，范围 0-100 |

示例：

```python
from src.indicators import (
    sma, ema, macd, rsi, bollinger_bands, kdj, atr,
    max_force_resonance, xpct_only,
)

df["ma20"] = sma(df, period=20)
df["ema12"] = ema(df, period=12)

macd_df = macd(df)
df["dif"] = macd_df["DIF"]
df["dea"] = macd_df["DEA"]
df["macd_hist"] = macd_df["MACD"]

df["rsi14"] = rsi(df, period=14)

bb = bollinger_bands(df, period=20, std_dev=2.0)
df["bb_middle"] = bb["middle"]
df["bb_upper"] = bb["upper"]
df["bb_lower"] = bb["lower"]

kdj_df = kdj(df)
df["k"] = kdj_df["K"]
df["d"] = kdj_df["D"]
df["j"] = kdj_df["J"]

df["atr14"] = atr(df, period=14)

main_force = max_force_resonance(df, n=12, m=240)
df["xpct"] = main_force["xpct"]
df["main_net_inflow"] = main_force["main_net_inflow"]
df["strong_buy"] = main_force["strong_buy"]
df["xg100_s"] = main_force["xg100_s"]
```

## 策略列表

内置策略都从 `src.strategies` 导入。除 `MainForceResonanceStrategy` 外，当前回测引擎期望策略信号格式为 `{"direction": "buy"|"sell", "amount": 股数}`。

| 策略 | 类 | 核心逻辑 | 主要参数 |
| --- | --- | --- | --- |
| 策略基类 | `BaseStrategy` | 自定义策略继承并实现 `on_bar(context)` | `name` |
| 买入持有 | `BuyAndHoldStrategy` | 首个 Bar 买入 100 股后持有 | 无 |
| 均线交叉 | `MovingAverageCrossStrategy` | 短均线上穿长均线买入，下穿卖出 | `short_window=5`, `long_window=20` |
| MACD | `MACDStrategy` | DIF 上穿 DEA 买入，下穿卖出 | `fast=12`, `slow=26`, `signal=9` |
| RSI | `RSIStrategy` | RSI 上穿超卖线买入，下穿超买线卖出 | `period=14`, `oversold=30`, `overbought=70` |
| 布林带 | `BollingerStrategy` | 价格上穿下轨买入，下穿上轨卖出 | `period=20`, `std_dev=2.0` |
| KDJ | `KDJStrategy` | K 上穿 D 且 J 低位买入，K 下穿 D 且 J 高位卖出 | `n=9`, `m1=3`, `m2=3`, `j_buy_threshold=20`, `j_sell_threshold=80` |
| ATR | `ATRStrategy` | 价格突破 ATR 通道上轨买入，跌破下轨卖出 | `atr_period=14`, `ma_period=20`, `multiplier=2.0`, `use_sma=True` |
| 主力共振 | `MainForceResonanceStrategy` | XPCT 低位和主力净流入共振买入，高位出货信号卖出 | `n=12`, `m=240`, `bp_buy=0`, `sp_sell=95`, `use_strong_buy_only=True` |

单策略示例：

```python
from src.data.manager import DataManager
from src.backtest.engine import BacktestEngine
from src.strategies import MACDStrategy

symbol = "000001.SZ"

df = DataManager().get_daily_data(symbol, "2023-01-01", "2024-12-31")
df.attrs["symbol"] = symbol

strategy = MACDStrategy(fast=12, slow=26, signal=9)
engine = BacktestEngine(initial_cash=100000, commission=0.0003)
result = engine.run(strategy=strategy, data=df)

print(result["total_return_pct"])
print(result["max_drawdown_pct"])
print(result["sharpe_ratio"])
print(result["total_trades"])
```

多策略对比示例：

```python
from src.backtest.engine import BacktestEngine
from src.strategies import (
    BuyAndHoldStrategy,
    MovingAverageCrossStrategy,
    MACDStrategy,
    RSIStrategy,
    BollingerStrategy,
    KDJStrategy,
    ATRStrategy,
)

strategies = {
    "买入持有": BuyAndHoldStrategy(),
    "均线交叉": MovingAverageCrossStrategy(short_window=5, long_window=20),
    "MACD": MACDStrategy(),
    "RSI": RSIStrategy(period=14, oversold=30, overbought=70),
    "布林带": BollingerStrategy(period=20, std_dev=2.0),
    "KDJ": KDJStrategy(),
    "ATR": ATRStrategy(),
}

results = {}
for name, strategy in strategies.items():
    engine = BacktestEngine(initial_cash=100000)
    results[name] = engine.run(strategy=strategy, data=df)

for name, result in sorted(results.items(), key=lambda item: item[1]["total_return"], reverse=True):
    print(name, result["total_return_pct"], result["total_trades"])
```

自定义策略示例：

```python
from src.strategies import BaseStrategy

class BreakoutStrategy(BaseStrategy):
    def __init__(self, window=20):
        super().__init__("BreakoutStrategy")
        self.window = window

    def on_bar(self, context):
        data = context["data"]
        if len(data) < self.window + 1:
            return None

        current_close = data["close"].iloc[-1]
        previous_high = data["high"].iloc[-self.window - 1:-1].max()
        previous_low = data["low"].iloc[-self.window - 1:-1].min()

        if current_close > previous_high:
            return {"direction": "buy", "amount": 100}
        if current_close < previous_low:
            return {"direction": "sell", "amount": 100}
        return None
```

## 回测执行

回测引擎输出字段：

| 字段 | 说明 |
| --- | --- |
| `initial_cash` | 初始资金 |
| `final_value` | 回测结束总资产 |
| `total_return` / `total_return_pct` | 总收益率 |
| `max_drawdown` / `max_drawdown_pct` | 最大回撤 |
| `sharpe_ratio` | 年化夏普比率，简化为无风险利率 0 |
| `total_trades` | 交易次数 |
| `daily_values` | 每日净值、现金和持仓市值 DataFrame |
| `trades` | 交易记录 DataFrame |

完整示例：

```python
from src.data.manager import DataManager
from src.data.storage import DataStorage
from src.backtest.engine import BacktestEngine
from src.strategies import RSIStrategy

symbol = "600519.SH"
start = "2022-01-01"
end = "2024-12-31"

dm = DataManager()
df = dm.get_daily_data(symbol, start=start, end=end, source="auto")
df.attrs["symbol"] = symbol

storage = DataStorage(backend="parquet")
storage.save_daily_data(symbol, df)

loaded = storage.load_daily_data(symbol, start="2023-01-01", end=end)
loaded.attrs["symbol"] = symbol

strategy = RSIStrategy(period=14, oversold=30, overbought=70)
engine = BacktestEngine(initial_cash=100000, commission=0.0003)
result = engine.run(strategy=strategy, data=loaded)

print(f"最终资产: {result['final_value']:,.2f}")
print(f"总收益率: {result['total_return_pct']}")
print(f"最大回撤: {result['max_drawdown_pct']}")
print(f"夏普比率: {result['sharpe_ratio']:.2f}")
print(f"交易次数: {result['total_trades']}")

print(result["trades"].tail())
```

也可以直接运行脚本：

```bash
python download_data.py --symbols 000001.SZ,600519.SH --start 2020-01-01 --end 2024-12-31
python quick_backtest.py
```

## 可视化

`src.visualization.charts` 提供三个入口：

| 函数 | 说明 |
| --- | --- |
| `plot_backtest_result(result, show_trades=True, backend="matplotlib")` | 绘制总资产、回撤、现金与持仓，可标注买卖点；`backend` 支持 `matplotlib` 和 `plotly` |
| `plot_price_with_signals(data, trades=None)` | 绘制价格走势和交易点 |
| `plot_strategy_comparison(results, labels)` | 对比多个策略的净值曲线、收益率、最大回撤和夏普比率 |

示例：

```python
from src.visualization.charts import (
    plot_backtest_result,
    plot_price_with_signals,
    plot_strategy_comparison,
)

# matplotlib 静态图
plot_backtest_result(result, show_trades=True, backend="matplotlib")

# plotly 交互图；如未安装，先执行 pip install plotly
plot_backtest_result(result, show_trades=True, backend="plotly")

# 价格和交易点
plot_price_with_signals(loaded, result["trades"])

# 多策略对比
plot_strategy_comparison(
    results=[results["买入持有"], results["MACD"], results["RSI"]],
    labels=["买入持有", "MACD", "RSI"],
)
```

## 数据格式约定

回测和指标默认使用日线数据：

```text
date, open, high, low, close, volume
```

注意事项：

- `BacktestEngine` 使用 `data.attrs["symbol"]` 作为交易标的；未设置时会显示为 `UNKNOWN`。
- `date` 建议使用可被 pandas 识别的日期格式。
- KDJ、ATR、主力共振等指标依赖 `high/low` 或成交量字段。
- `MainForceResonanceStrategy.on_bar()` 当前返回 `action/amount` 格式，适合直接调用 `generate_signals()` 查看详细信号；如接入 `BacktestEngine`，需要和引擎的 `direction/amount` 信号格式对齐。

## 测试

```bash
pytest
```

或运行指标测试：

```bash
python -m unittest tests/test_indicators.py
```

## 许可证

MIT
