"""
LON 龙系长线策略 —— 15 分钟多空双向回测

在上一版「全仓多 / 空仓」基础上，新增「做空」能力（指标反向对称）：

  - LON 在 0 轴下方，柱体长度连续两天变短 -> 做多（空头能量衰竭，潜在底部）
  - LON 在 0 轴上方，柱体长度连续两天变短 -> 做空（多头能量衰竭，潜在顶部）

对比两种执行方式：
  - long_only ：买入信号满仓做多，卖出信号清仓（原版）
  - long_short：买入信号做多，卖出信号反手做空（本次新增）

只测 15 分钟周期，覆盖多类品种（美股 / 指数 / 加密货币 / 期货）。
LON 是量价指标，无成交量的品种（如外汇、连续合约日线）会退化，脚本自动跳过。

回测采用收益率法（持仓 ×bar 收益 − 换手手续费），可正确处理做空与多空反手。
"""

import sys
import os
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.manager import YahooFinanceDataSource
from src.strategies.lon_strategy import LONStrategy


# 不同类型品种（Yahoo 代码），仅 15 分钟
INSTRUMENTS = [
    # 美股龙头
    ("苹果 AAPL",   "AAPL",    "美股"),
    ("英伟达 NVDA", "NVDA",    "美股"),
    ("特斯拉 TSLA", "TSLA",    "美股"),
    ("亚马逊 AMZN", "AMZN",    "美股"),
    # 股指
    ("标普500",     "^GSPC",   "指数"),
    ("纳斯达克",    "^IXIC",   "指数"),
    ("上证指数",    "000001.SS", "指数"),
    ("恒生指数",    "^HSI",    "指数"),
    # 加密货币
    ("比特币 BTC",  "BTC-USD", "加密货币"),
    ("以太坊 ETH",  "ETH-USD", "加密货币"),
    ("Solana SOL",  "SOL-USD", "加密货币"),
    ("狗狗币 DOGE", "DOGE-USD", "加密货币"),
    # 期货
    ("标普期货 ES", "ES=F",    "期货"),
    ("纳指期货 NQ", "NQ=F",    "期货"),
    ("原油 CL",     "CL=F",    "期货"),
    ("天然气 NG",   "NG=F",    "期货"),
    ("黄金 GC",     "GC=F",    "期货"),
    ("铜 HG",       "HG=F",    "期货"),
]

RANGE = "60d"
INTERVAL = "15m"
PERIODS_PER_YEAR = 252 * 26  # 15 分钟年化交易周期数（美股每日约 26 根）


@dataclass
class Result:
    name: str
    category: str
    bars: int
    bh_return: float
    long_only_return: float
    long_short_return: float
    lo_drawdown: float
    ls_drawdown: float
    lo_sharpe: float
    ls_sharpe: float
    ls_trades: int
    ls_win_rate: float


def _build_positions(strategy: LONStrategy, df: pd.DataFrame, mode: str) -> np.ndarray:
    """逐 bar 调用策略信号，构造每根 K 线收盘后持有的目标仓位 (+1/-1/0)。"""
    n = len(df)
    closes = df['close'].values
    targets = np.zeros(n)
    cur = 0
    for i in range(n):
        window = df.iloc[: i + 1]
        sig = strategy.on_bar({
            'data': window,
            'price': float(closes[i]),
            'date': df['date'].iloc[i] if 'date' in df.columns else i,
            'portfolio': None,
        })
        d = sig.get('direction') if sig else None
        if d == 'buy':
            cur = 1
        elif d == 'sell':
            cur = -1 if mode == 'long_short' else 0
        targets[i] = cur
    return targets


def _simulate(targets: np.ndarray, closes: np.ndarray,
              commission: float = 0.0003) -> Dict[str, np.ndarray]:
    """收益率法回测：第 i 根 bar 的收益由上一根收盘确定的仓位获得，换手计手续费。"""
    n = len(closes)
    bar_ret = np.zeros(n)
    bar_ret[1:] = closes[1:] / closes[:-1] - 1.0

    strat_ret = np.zeros(n)
    strat_ret[1:] = targets[:-1] * bar_ret[1:]

    turnover = np.zeros(n)
    turnover[0] = abs(targets[0])
    turnover[1:] = np.abs(np.diff(targets))
    strat_ret = strat_ret - turnover * commission

    equity = np.cumprod(1.0 + strat_ret)
    return {'ret': strat_ret, 'equity': equity}


def _metrics(strat_ret: np.ndarray, equity: np.ndarray, ppy: int):
    total = float(equity[-1] - 1.0)
    cummax = np.maximum.accumulate(equity)
    dd = float(((equity - cummax) / cummax).min())
    r = strat_ret[1:]
    sharpe = float(r.mean() / r.std() * math.sqrt(ppy)) if r.std() > 0 else 0.0
    return total, dd, sharpe


def _trade_stats(targets: np.ndarray, equity: np.ndarray):
    """把"持有非零仓位"的连续区段视为一笔交易，统计笔数与胜率。"""
    n = len(targets)
    trades = []
    seg_start = None
    for i in range(n):
        if targets[i] != 0 and (i == 0 or targets[i] != targets[i - 1]):
            seg_start = i
        # 区段结束：仓位改变 或 到末尾
        end = (i == n - 1) or (targets[i] != 0 and targets[i + 1] != targets[i])
        if seg_start is not None and targets[i] != 0 and end:
            # 该段累计收益 = equity[end]/equity[seg_start-1] - 1
            base = equity[seg_start - 1] if seg_start > 0 else 1.0
            seg_ret = equity[i] / base - 1.0
            trades.append(seg_ret)
            seg_start = None
    wins = sum(1 for t in trades if t > 0)
    win_rate = wins / len(trades) if trades else 0.0
    return len(trades), win_rate


def run() -> List[Result]:
    src = YahooFinanceDataSource()
    results: List[Result] = []

    print(f"\n{'='*108}")
    print(f"LON 多空双向回测  周期=15分钟 (range={RANGE})  手续费=万3")
    print(f"{'='*108}")
    print(f"{'品种':<13}{'类别':<10}{'K线':>6}{'买入持有':>10}"
          f"{'纯做多':>10}{'多空双向':>11}{'多空回撤':>10}{'多空夏普':>9}{'交易':>5}{'胜率':>7}")
    print("-" * 108)

    for disp, sym, cat in INSTRUMENTS:
        df = src.get_kline(sym, range_=RANGE, interval=INTERVAL)
        if df is None or len(df) < 60 or df['volume'].sum() <= 0:
            print(f"{disp:<13}{cat:<10}{'无数据/无成交量(跳过)':>40}")
            continue

        closes = df['close'].values

        lo_pos = _build_positions(LONStrategy(), df, 'long_only')
        ls_pos = _build_positions(LONStrategy(), df, 'long_short')
        lo = _simulate(lo_pos, closes)
        ls = _simulate(ls_pos, closes)

        lo_total, lo_dd, lo_sharpe = _metrics(lo['ret'], lo['equity'], PERIODS_PER_YEAR)
        ls_total, ls_dd, ls_sharpe = _metrics(ls['ret'], ls['equity'], PERIODS_PER_YEAR)
        ls_trades, ls_wr = _trade_stats(ls_pos, ls['equity'])
        bh = float(closes[-1] / closes[0] - 1.0)

        results.append(Result(
            name=disp, category=cat, bars=len(df), bh_return=bh,
            long_only_return=lo_total, long_short_return=ls_total,
            lo_drawdown=lo_dd, ls_drawdown=ls_dd,
            lo_sharpe=lo_sharpe, ls_sharpe=ls_sharpe,
            ls_trades=ls_trades, ls_win_rate=ls_wr,
        ))

        print(f"{disp:<13}{cat:<10}{len(df):>6}{bh*100:>9.1f}%"
              f"{lo_total*100:>9.1f}%{ls_total*100:>10.1f}%"
              f"{ls_dd*100:>9.1f}%{ls_sharpe:>9.2f}{ls_trades:>5}{ls_wr*100:>6.1f}%")

    _summary(results)
    return results


def _summary(results: List[Result]) -> None:
    if not results:
        return
    print("-" * 108)
    bh = np.array([r.bh_return for r in results])
    lo = np.array([r.long_only_return for r in results])
    ls = np.array([r.long_short_return for r in results])
    print(f"总体({len(results)}品种)："
          f" 买入持有均 {bh.mean()*100:+.1f}% | 纯做多均 {lo.mean()*100:+.1f}% |"
          f" 多空双向均 {ls.mean()*100:+.1f}% | 多空正收益 {int((ls>0).sum())}/{len(results)} |"
          f" 多空跑赢B&H {int((ls>bh).sum())}/{len(results)} | 多空优于纯做多 {int((ls>lo).sum())}/{len(results)}")

    # 分类别
    cats = {}
    for r in results:
        cats.setdefault(r.category, []).append(r)
    print("\n按类别（多空双向平均收益 / 平均夏普 / 平均胜率）：")
    for cat, rs in cats.items():
        m = np.mean([r.long_short_return for r in rs])
        s = np.mean([r.ls_sharpe for r in rs])
        w = np.mean([r.ls_win_rate for r in rs])
        print(f"  {cat:<8} {m*100:+7.1f}%   夏普 {s:5.2f}   胜率 {w*100:4.0f}%   ({len(rs)}个)")


def write_report(results: List[Result], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    L: List[str] = []
    L.append("# LON 龙系长线策略 —— 15 分钟多空双向回测\n\n")
    L.append("## 策略与执行\n\n")
    L.append("信号（指标反向对称）：\n")
    L.append("- LON 0 轴**下方**柱体连续两天变短 -> **做多**（空头能量衰竭）\n")
    L.append("- LON 0 轴**上方**柱体连续两天变短 -> **做空**（多头能量衰竭）\n\n")
    L.append("两种执行对比：**纯做多**（卖出信号清仓）vs **多空双向**（卖出信号反手做空）。\n\n")
    L.append(f"周期 15 分钟（约 60 天），手续费万分之三，收益率法回测。数据：Yahoo Finance。\n\n")

    L.append("## 明细\n\n")
    L.append("| 品种 | 类别 | K线 | 买入持有 | 纯做多 | 多空双向 | 多空回撤 | 多空夏普 | 交易 | 胜率 |\n")
    L.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for r in results:
        L.append(f"| {r.name} | {r.category} | {r.bars} | {r.bh_return*100:+.1f}% | "
                 f"{r.long_only_return*100:+.1f}% | {r.long_short_return*100:+.1f}% | "
                 f"{r.ls_drawdown*100:.1f}% | {r.ls_sharpe:.2f} | {r.ls_trades} | {r.ls_win_rate*100:.0f}% |\n")

    bh = np.array([r.bh_return for r in results])
    lo = np.array([r.long_only_return for r in results])
    ls = np.array([r.long_short_return for r in results])
    L.append(f"\n**总体（{len(results)} 品种）**：买入持有均 {bh.mean()*100:+.1f}%，纯做多均 {lo.mean()*100:+.1f}%，"
             f"多空双向均 {ls.mean()*100:+.1f}%；多空正收益 {int((ls>0).sum())}/{len(results)}，"
             f"多空跑赢买入持有 {int((ls>bh).sum())}/{len(results)}，多空优于纯做多 {int((ls>lo).sum())}/{len(results)}。\n\n")

    cats = {}
    for r in results:
        cats.setdefault(r.category, []).append(r)
    L.append("## 按品种类别（多空双向）\n\n")
    L.append("| 类别 | 平均收益 | 平均夏普 | 平均胜率 | 品种数 |\n")
    L.append("| --- | ---: | ---: | ---: | ---: |\n")
    for cat, rs in cats.items():
        m = np.mean([r.long_short_return for r in rs])
        s = np.mean([r.ls_sharpe for r in rs])
        w = np.mean([r.ls_win_rate for r in rs])
        L.append(f"| {cat} | {m*100:+.1f}% | {s:.2f} | {w*100:.0f}% | {len(rs)} |\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(L))
    print(f"\n📄 报告已写入: {path}")


def main():
    results = run()
    report = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "results", "lon_long_short_report.md")
    write_report(results, report)


if __name__ == "__main__":
    main()
