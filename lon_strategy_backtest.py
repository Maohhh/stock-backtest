"""
LON 龙系长线策略 —— 跨品种 / 跨周期回测

策略思想（以 LON 长线指标为基础）：
  - 在 LON 0 轴下方，柱体长度连续两天变短时买入（空头能量衰竭）
  - 在 LON 0 轴上方，柱体长度连续两天变短时卖出（多头能量衰竭）

本脚本在多类品种（美股 / 指数 / 加密货币 / 期货 / 外汇 / 中国指数）上，
分别用「日线」与「15 分钟」两个周期回测该策略，并与买入持有(Buy&Hold)对比。

数据来源：Yahoo Finance 公开 chart 接口（HTTP，可跨市场、跨周期）。
由于不同品种价格量级差异巨大，这里采用「全仓多/空仓」方式模拟：
买入信号满仓做多，卖出信号清仓离场，从而得到与价格量级无关的可比收益率。
"""

import sys
import os
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.manager import YahooFinanceDataSource
from src.strategies.lon_strategy import LONStrategy


# 测试品种：覆盖股票 / 指数 / 加密货币 / 期货 / 外汇 / 中国指数
INSTRUMENTS = [
    # (展示名称, Yahoo 代码, 类别)
    ("苹果 AAPL",      "AAPL",       "美股"),
    ("微软 MSFT",      "MSFT",       "美股"),
    ("英伟达 NVDA",    "NVDA",       "美股"),
    ("特斯拉 TSLA",    "TSLA",       "美股"),
    ("标普500",        "^GSPC",      "指数"),
    ("纳斯达克",       "^IXIC",      "指数"),
    ("上证指数",       "000001.SS",  "中国指数"),
    ("比特币 BTC",     "BTC-USD",    "加密货币"),
    ("以太坊 ETH",     "ETH-USD",    "加密货币"),
    ("黄金期货 GC",    "GC=F",       "期货"),
    ("原油期货 CL",    "CL=F",       "期货"),
    ("欧元美元",       "EURUSD=X",   "外汇"),
]

# 每个周期对应的 (range, interval, 年化交易周期数)
TIMEFRAMES = {
    "日线":   ("2y", "1d", 252),
    "15分钟": ("60d", "15m", 252 * 26),  # 美股每日约 26 根 15 分钟K线
}


@dataclass
class TradeStat:
    entry_price: float
    exit_price: float

    @property
    def ret(self) -> float:
        return self.exit_price / self.entry_price - 1.0


@dataclass
class BacktestResult:
    name: str
    category: str
    bars: int
    strat_return: float
    bh_return: float
    max_drawdown: float
    sharpe: float
    num_trades: int
    win_rate: float
    final_pos_open: bool


def simulate_long_flat(
    strategy: LONStrategy,
    df: pd.DataFrame,
    periods_per_year: int,
    initial_cash: float = 100_000.0,
    commission: float = 0.0003,
) -> BacktestResult:
    """
    用「全仓多 / 空仓」方式驱动真实的 LONStrategy 信号进行回测。

    - 买入信号 & 当前空仓 -> 满仓做多（允许小数股，便于跨品种比较）
    - 卖出信号 & 当前持仓 -> 全部卖出离场
    """
    cash = initial_cash
    shares = 0.0
    in_market = False
    entry_price = 0.0

    equity_curve: List[float] = []
    trades: List[TradeStat] = []

    closes = df['close'].values

    for i in range(len(df)):
        price = float(closes[i])
        window = df.iloc[: i + 1]
        context = {
            'date': df['date'].iloc[i] if 'date' in df.columns else i,
            'price': price,
            'portfolio': None,
            'data': window,
        }
        signal = strategy.on_bar(context)
        direction = signal.get('direction') if signal else None

        if direction == 'buy' and not in_market:
            shares = cash / (price * (1 + commission))
            cash = 0.0
            in_market = True
            entry_price = price
        elif direction == 'sell' and in_market:
            cash = shares * price * (1 - commission)
            trades.append(TradeStat(entry_price=entry_price, exit_price=price))
            shares = 0.0
            in_market = False

        equity_curve.append(cash + shares * price)

    equity = pd.Series(equity_curve)
    strat_return = equity.iloc[-1] / initial_cash - 1.0
    bh_return = float(closes[-1]) / float(closes[0]) - 1.0

    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_dd = float(drawdown.min())

    rets = equity.pct_change().dropna()
    if rets.std() > 0:
        sharpe = float(rets.mean() / rets.std() * math.sqrt(periods_per_year))
    else:
        sharpe = 0.0

    wins = sum(1 for t in trades if t.ret > 0)
    win_rate = wins / len(trades) if trades else 0.0

    return BacktestResult(
        name=strategy.name,
        category="",
        bars=len(df),
        strat_return=strat_return,
        bh_return=bh_return,
        max_drawdown=max_dd,
        sharpe=sharpe,
        num_trades=len(trades),
        win_rate=win_rate,
        final_pos_open=in_market,
    )


def run_all() -> Dict[str, List[BacktestResult]]:
    source = YahooFinanceDataSource()
    all_results: Dict[str, List[BacktestResult]] = {tf: [] for tf in TIMEFRAMES}

    for tf_name, (range_, interval, ppy) in TIMEFRAMES.items():
        print("\n" + "=" * 92)
        print(f"周期：{tf_name}  (range={range_}, interval={interval})")
        print("=" * 92)
        header = f"{'品种':<14}{'类别':<10}{'K线数':>7}{'策略收益':>11}{'买入持有':>11}{'超额':>10}{'最大回撤':>10}{'夏普':>8}{'交易数':>7}{'胜率':>8}"
        print(header)
        print("-" * 92)

        for disp_name, symbol, category in INSTRUMENTS:
            df = source.get_kline(symbol, range_=range_, interval=interval)
            if df is None or len(df) < 60:
                print(f"{disp_name:<14}{category:<10}{'无数据/不足':>40}")
                continue

            strat = LONStrategy()
            res = simulate_long_flat(strat, df, periods_per_year=ppy)
            res.name = disp_name
            res.category = category
            all_results[tf_name].append(res)

            excess = res.strat_return - res.bh_return
            print(f"{disp_name:<14}{category:<10}{res.bars:>7}"
                  f"{res.strat_return*100:>10.2f}%"
                  f"{res.bh_return*100:>10.2f}%"
                  f"{excess*100:>9.2f}%"
                  f"{res.max_drawdown*100:>9.2f}%"
                  f"{res.sharpe:>8.2f}"
                  f"{res.num_trades:>7}"
                  f"{res.win_rate*100:>7.1f}%")

        _print_summary(all_results[tf_name])

    return all_results


def _print_summary(results: List[BacktestResult]) -> None:
    if not results:
        return
    strat = np.array([r.strat_return for r in results])
    bh = np.array([r.bh_return for r in results])
    excess = strat - bh
    print("-" * 92)
    print(f"汇总({len(results)}个品种)："
          f" 策略平均 {strat.mean()*100:+.2f}%  |"
          f" 买入持有平均 {bh.mean()*100:+.2f}%  |"
          f" 平均超额 {excess.mean()*100:+.2f}%  |"
          f" 跑赢B&H {int((excess > 0).sum())}/{len(results)}  |"
          f" 正收益 {int((strat > 0).sum())}/{len(results)}  |"
          f" 平均胜率 {np.mean([r.win_rate for r in results])*100:.1f}%")


def write_report(all_results: Dict[str, List[BacktestResult]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines: List[str] = []
    lines.append("# LON 龙系长线策略 —— 跨品种 / 跨周期回测报告\n")
    lines.append("## 策略规则\n")
    lines.append("以 LON 长线指标为基础：\n")
    lines.append("- **买入**：LON 在 0 轴下方（空头能量区），且 LON 柱体长度连续两天变短。\n")
    lines.append("- **卖出**：LON 在 0 轴上方（多头能量区），且 LON 柱体长度连续两天变短。\n")
    lines.append("\n回测采用「全仓多 / 空仓」模拟，手续费万分之三，初始资金 10 万。\n")
    lines.append("数据来源：Yahoo Finance（日线 2 年，15 分钟约 60 天）。\n")

    for tf_name, results in all_results.items():
        lines.append(f"\n## {tf_name}\n")
        if not results:
            lines.append("\n（无可用数据）\n")
            continue
        lines.append("\n| 品种 | 类别 | K线数 | 策略收益 | 买入持有 | 超额 | 最大回撤 | 夏普 | 交易数 | 胜率 |")
        lines.append("\n| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for r in results:
            excess = r.strat_return - r.bh_return
            lines.append(
                f"| {r.name} | {r.category} | {r.bars} | "
                f"{r.strat_return*100:+.2f}% | {r.bh_return*100:+.2f}% | {excess*100:+.2f}% | "
                f"{r.max_drawdown*100:.2f}% | {r.sharpe:.2f} | {r.num_trades} | {r.win_rate*100:.1f}% |\n"
            )
        strat = np.array([r.strat_return for r in results])
        bh = np.array([r.bh_return for r in results])
        excess = strat - bh
        lines.append(
            f"\n**汇总**：策略平均 {strat.mean()*100:+.2f}%，买入持有平均 {bh.mean()*100:+.2f}%，"
            f"平均超额 {excess.mean()*100:+.2f}%，跑赢买入持有 {int((excess>0).sum())}/{len(results)}，"
            f"正收益 {int((strat>0).sum())}/{len(results)}，平均胜率 {np.mean([r.win_rate for r in results])*100:.1f}%。\n"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    print(f"\n📄 报告已写入: {path}")


def main():
    print("LON 龙系长线策略 —— 跨品种 / 跨周期回测")
    results = run_all()
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "results", "lon_strategy_report.md")
    write_report(results, report_path)


if __name__ == "__main__":
    main()
