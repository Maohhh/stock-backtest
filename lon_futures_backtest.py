"""
LON 龙系长线策略 —— 国内期货全品种 15 分钟回测

数据：仓库 `data_futures/15min/*.parquet`（国内期货持仓量加权连续 15 分钟 K 线，
约 2023-09 ~ 2026-05，含 volume / open_interest，LON 量价指标不会退化）。

策略（指标反向对称）：
  - LON 0 轴下方柱体长度连续两天变短 -> 做多（空头能量衰竭）
  - LON 0 轴上方柱体长度连续两天变短 -> 做空（多头能量衰竭）

对比两种执行：
  - 纯做多 long_only ：卖出信号清仓
  - 多空双向 long_short：卖出信号反手做空

LON 信号为因果计算（rolling/cumsum/ewm，无未来函数），故按整段序列向量化，
其结果与逐 bar 调用 LONStrategy.on_bar 完全一致（已验证）。
"""

import os
import sys
import math
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.indicators.lon import lon
from lon_long_short_backtest import _simulate, _metrics, _trade_stats

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_futures", "15min")
MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_futures", "manifest_15min.csv")
PERIODS_PER_YEAR = 252 * 16  # 国内期货含夜盘，每日约 16 根 15 分钟K线
COMMISSION = 0.0003
DEA_PERIOD = 20


@dataclass
class Row:
    product: str
    name: str
    exchange: str
    bars: int
    bh: float
    lo_ret: float
    ls_ret: float
    lo_dd: float
    ls_dd: float
    lo_sharpe: float
    ls_sharpe: float
    ls_trades: int
    ls_win: float


def lon_signals(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """向量化 LON 买/卖信号（与 LONStrategy.on_bar 等价）。"""
    ld = lon(df)
    lv = ld['lon'].values
    length = np.abs(lv)
    n = len(df)
    shrink2 = np.zeros(n, dtype=bool)
    shrink2[2:] = (length[2:] < length[1:-1]) & (length[1:-1] < length[:-2])
    buy = shrink2 & (lv < 0)
    sell = shrink2 & (lv > 0)
    warm = max(DEA_PERIOD, 3) + 3  # 与策略预热一致
    buy[: warm - 1] = False
    sell[: warm - 1] = False
    return {'buy': buy, 'sell': sell}


def build_targets(buy: np.ndarray, sell: np.ndarray, mode: str) -> np.ndarray:
    n = len(buy)
    targets = np.zeros(n)
    cur = 0
    for i in range(n):
        if buy[i]:
            cur = 1
        elif sell[i]:
            cur = -1 if mode == 'long_short' else 0
        targets[i] = cur
    return targets


def backtest_product(df: pd.DataFrame):
    closes = df['close'].values
    sig = lon_signals(df)
    out = {}
    for mode in ('long_only', 'long_short'):
        tgt = build_targets(sig['buy'], sig['sell'], mode)
        sim = _simulate(tgt, closes, commission=COMMISSION)
        total, dd, sharpe = _metrics(sim['ret'], sim['equity'], PERIODS_PER_YEAR)
        trades, win = _trade_stats(tgt, sim['equity'])
        out[mode] = (total, dd, sharpe, trades, win)
    bh = float(closes[-1] / closes[0] - 1.0)
    return bh, out


def _ensure_data() -> None:
    """数据存放在 futures-weighted-data-download 分支，本分支未重复提交，缺失时给出取数命令。"""
    if os.path.isdir(DATA_DIR) and any(f.endswith('.parquet') for f in os.listdir(DATA_DIR)):
        return
    sys.exit(
        "未找到期货数据目录 data_futures/15min/。\n"
        "请先从数据分支取数后再运行：\n"
        "  git checkout origin/claude/futures-weighted-data-download-LRsDy -- "
        "data_futures/15min data_futures/manifest_15min.csv\n"
        "  python3 lon_futures_backtest.py"
    )


def run() -> List[Row]:
    _ensure_data()
    man = pd.read_csv(MANIFEST).set_index('product')
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith('.parquet'))
    rows: List[Row] = []

    for f in files:
        product = f[:-len('.parquet')]
        df = pd.read_parquet(os.path.join(DATA_DIR, f)).rename(columns={'datetime': 'date'})
        if len(df) < 100 or df['volume'].sum() <= 0:
            continue
        bh, out = backtest_product(df)
        lo = out['long_only']
        ls = out['long_short']
        info = man.loc[product] if product in man.index else None
        rows.append(Row(
            product=product,
            name=str(info['name']) if info is not None else product,
            exchange=str(info['exchange']) if info is not None else '?',
            bars=len(df), bh=bh,
            lo_ret=lo[0], ls_ret=ls[0], lo_dd=lo[1], ls_dd=ls[1],
            lo_sharpe=lo[2], ls_sharpe=ls[2], ls_trades=ls[3], ls_win=ls[4],
        ))
    return rows


def _agg(label: str, rows: List[Row]) -> str:
    bh = np.array([r.bh for r in rows])
    lo = np.array([r.lo_ret for r in rows])
    ls = np.array([r.ls_ret for r in rows])
    return (f"{label}({len(rows)}): 买入持有均 {bh.mean()*100:+.1f}% | 纯做多均 {lo.mean()*100:+.1f}% "
            f"| 多空均 {ls.mean()*100:+.1f}% | 纯做多正收益 {int((lo>0).sum())}/{len(rows)} "
            f"| 多空正收益 {int((ls>0).sum())}/{len(rows)} | 多空胜率均 {np.mean([r.ls_win for r in rows])*100:.0f}%")


def report(rows: List[Row]) -> None:
    rows_sorted = sorted(rows, key=lambda r: r.ls_ret, reverse=True)

    print(f"\n{'='*112}")
    print(f"LON 期货全品种 15 分钟回测  ({len(rows)} 个品种, 数据~2023-09~2026-05, 手续费万3)")
    print(f"{'='*112}")
    print(f"{'品种':<6}{'名称':<8}{'所':<6}{'K线':>7}{'买入持有':>10}{'纯做多':>10}{'多空双向':>11}{'多空回撤':>10}{'多空夏普':>9}{'交易':>5}{'胜率':>7}")
    print("-" * 112)
    for r in rows_sorted:
        print(f"{r.product:<6}{r.name[:7]:<8}{r.exchange:<6}{r.bars:>7}{r.bh*100:>9.1f}%"
              f"{r.lo_ret*100:>9.1f}%{r.ls_ret*100:>10.1f}%{r.ls_dd*100:>9.1f}%"
              f"{r.ls_sharpe:>9.2f}{r.ls_trades:>5}{r.ls_win*100:>6.0f}%")
    print("-" * 112)
    print(_agg("总体", rows))
    # 按交易所
    ex: Dict[str, List[Row]] = {}
    for r in rows:
        ex.setdefault(r.exchange, []).append(r)
    print("\n按交易所：")
    for k, v in sorted(ex.items()):
        print("  " + _agg(k, v))


def write_md(rows: List[Row], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows_sorted = sorted(rows, key=lambda r: r.ls_ret, reverse=True)
    bh = np.array([r.bh for r in rows]); lo = np.array([r.lo_ret for r in rows]); ls = np.array([r.ls_ret for r in rows])
    L: List[str] = []
    L.append("# LON 龙系长线策略 —— 国内期货全品种 15 分钟回测\n\n")
    L.append("数据：仓库 `data_futures/15min/`（国内期货持仓量加权连续 15 分钟，约 2023-09 ~ 2026-05，"
             "含成交量/持仓量）。手续费万分之三，收益率法回测。\n\n")
    L.append("信号（指标反向对称）：LON 0 轴下方柱体连续两天变短做多，0 轴上方连续两天变短做空。"
             "对比**纯做多**（卖出清仓）与**多空双向**（卖出反手做空）。\n\n")
    L.append(f"**总体（{len(rows)} 品种）**：买入持有均 {bh.mean()*100:+.1f}%，"
             f"纯做多均 {lo.mean()*100:+.1f}%，多空双向均 {ls.mean()*100:+.1f}%；"
             f"纯做多正收益 {int((lo>0).sum())}/{len(rows)}，多空正收益 {int((ls>0).sum())}/{len(rows)}，"
             f"多空优于纯做多 {int((ls>lo).sum())}/{len(rows)}，多空跑赢买入持有 {int((ls>bh).sum())}/{len(rows)}。\n\n")

    # 按交易所
    ex: Dict[str, List[Row]] = {}
    for r in rows:
        ex.setdefault(r.exchange, []).append(r)
    L.append("## 按交易所（多空双向）\n\n| 交易所 | 品种数 | 买入持有均 | 纯做多均 | 多空双向均 | 多空夏普均 | 多空胜率均 |\n")
    L.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for k, v in sorted(ex.items()):
        L.append(f"| {k} | {len(v)} | {np.mean([r.bh for r in v])*100:+.1f}% | "
                 f"{np.mean([r.lo_ret for r in v])*100:+.1f}% | {np.mean([r.ls_ret for r in v])*100:+.1f}% | "
                 f"{np.mean([r.ls_sharpe for r in v]):.2f} | {np.mean([r.ls_win for r in v])*100:.0f}% |\n")

    L.append("\n## 全品种明细（按多空双向收益降序）\n\n")
    L.append("| 品种 | 名称 | 所 | K线 | 买入持有 | 纯做多 | 多空双向 | 多空回撤 | 多空夏普 | 交易 | 胜率 |\n")
    L.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for r in rows_sorted:
        L.append(f"| {r.product} | {r.name} | {r.exchange} | {r.bars} | {r.bh*100:+.1f}% | "
                 f"{r.lo_ret*100:+.1f}% | {r.ls_ret*100:+.1f}% | {r.ls_dd*100:.1f}% | "
                 f"{r.ls_sharpe:.2f} | {r.ls_trades} | {r.ls_win*100:.0f}% |\n")

    with open(path, "w", encoding="utf-8") as fp:
        fp.write("".join(L))
    print(f"\n📄 报告已写入: {path}")


def main():
    rows = run()
    report(rows)
    write_md(rows, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "results", "lon_futures_report.md"))


if __name__ == "__main__":
    main()
