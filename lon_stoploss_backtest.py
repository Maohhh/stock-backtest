"""
LON 龙系长线策略 —— 带止损/止盈的状态机版（仅用 LON 指标）

规则（用户设计）：
  做多：LON 0 轴下方(绿柱)，柱体长度连续两根变短 -> 开多
        持多且仍在 0 轴下方时，绿柱长度变长 1 根 -> 止损平仓
        多单反转到 0 轴上方(红柱)、红柱开始变短 -> 止盈平仓
  做空：LON 0 轴上方(红柱)，柱体长度连续两根变短 -> 开空
        持空且仍在 0 轴上方时，红柱长度变长 1 根 -> 止损平仓
        空单反转到 0 轴下方(绿柱)、绿柱开始变短 -> 止盈平仓

与"始终在场反手"版的区别：本版**有止损、大部分时间空仓**，止盈在反向动能见顶时离场。
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.indicators.lon import lon
from lon_futures_backtest import (
    DATA_DIR, MANIFEST, PERIODS_PER_YEAR, COMMISSION, DEA_PERIOD,
)
from lon_long_short_backtest import _simulate, _metrics, _trade_stats


def build_targets_sm(df: pd.DataFrame) -> np.ndarray:
    """按状态机逐 bar 生成目标仓位 (+1 多 / -1 空 / 0 空仓)。"""
    ld = lon(df)
    L = ld['lon'].values
    length = np.abs(L)
    n = len(df)
    warm = max(DEA_PERIOD, 3) + 3

    targets = np.zeros(n)
    pos = 0
    for t in range(n):
        if t < 2:
            targets[t] = 0
            continue
        len0, len1, len2 = length[t], length[t - 1], length[t - 2]
        grow = len0 > len1            # 当前柱比上一根变长
        shrink = len0 < len1          # 变短
        shrink2 = shrink and (len1 < len2)  # 连续两根变短
        lon_t = L[t]

        # 先处理离场（止损/止盈）
        if pos == 1:
            if lon_t < 0 and grow:        # 多单：仍在 0 轴下方且绿柱变长 -> 止损
                pos = 0
            elif lon_t > 0 and shrink:    # 多单：已到 0 轴上方且红柱见顶变短 -> 止盈
                pos = 0
        elif pos == -1:
            if lon_t > 0 and grow:        # 空单：仍在 0 轴上方且红柱变长 -> 止损
                pos = 0
            elif lon_t < 0 and shrink:    # 空单：已到 0 轴下方且绿柱见底变短 -> 止盈
                pos = 0

        # 空仓则按开仓条件进场（允许同根 K 线离场后反向开仓）
        if pos == 0 and t >= warm - 1:
            if lon_t < 0 and shrink2:
                pos = 1
            elif lon_t > 0 and shrink2:
                pos = -1

        targets[t] = pos
    return targets


@dataclass
class Row:
    product: str
    name: str
    exchange: str
    bars: int
    bh: float
    sm_ret: float
    sm_dd: float
    sm_sharpe: float
    sm_trades: int
    sm_win: float
    sm_exposure: float  # 在场时间占比


def run() -> List[Row]:
    man = pd.read_csv(MANIFEST).set_index('product')
    rows: List[Row] = []
    for f in sorted(os.listdir(DATA_DIR)):
        if not f.endswith('.parquet'):
            continue
        product = f[:-len('.parquet')]
        df = pd.read_parquet(os.path.join(DATA_DIR, f)).rename(columns={'datetime': 'date'})
        if len(df) < 100 or df['volume'].sum() <= 0:
            continue
        closes = df['close'].values
        tgt = build_targets_sm(df)
        sim = _simulate(tgt, closes, commission=COMMISSION)
        total, dd, sharpe = _metrics(sim['ret'], sim['equity'], PERIODS_PER_YEAR)
        trades, win = _trade_stats(tgt, sim['equity'])
        bh = float(closes[-1] / closes[0] - 1.0)
        info = man.loc[product] if product in man.index else None
        rows.append(Row(
            product=product,
            name=str(info['name']) if info is not None else product,
            exchange=str(info['exchange']) if info is not None else '?',
            bars=len(df), bh=bh, sm_ret=total, sm_dd=dd, sm_sharpe=sharpe,
            sm_trades=trades, sm_win=win, sm_exposure=float((tgt != 0).mean()),
        ))
    return rows


def report(rows: List[Row]) -> None:
    rs = sorted(rows, key=lambda r: r.sm_ret, reverse=True)
    print(f"\n{'='*104}")
    print(f"LON 止损止盈状态机版  期货 15 分钟  ({len(rows)} 品种)")
    print(f"{'='*104}")
    print(f"{'品种':<5}{'名称':<8}{'所':<6}{'买入持有':>10}{'本版收益':>10}{'回撤':>9}{'夏普':>7}{'交易':>5}{'胜率':>6}{'在场%':>7}")
    print("-" * 104)
    for r in rs:
        print(f"{r.product:<5}{r.name[:7]:<8}{r.exchange:<6}{r.bh*100:>9.1f}%{r.sm_ret*100:>9.1f}%"
              f"{r.sm_dd*100:>8.1f}%{r.sm_sharpe:>7.2f}{r.sm_trades:>5}{r.sm_win*100:>5.0f}%{r.sm_exposure*100:>6.0f}%")
    print("-" * 104)
    sm = np.array([r.sm_ret for r in rows])
    bh = np.array([r.bh for r in rows])
    print(f"总体({len(rows)})：买入持有均 {bh.mean()*100:+.1f}% | 本版均 {sm.mean()*100:+.1f}% | "
          f"本版正收益 {int((sm>0).sum())}/{len(rows)} | 跑赢买入持有 {int((sm>bh).sum())}/{len(rows)} | "
          f"平均胜率 {np.mean([r.sm_win for r in rows])*100:.0f}% | 平均在场 {np.mean([r.sm_exposure for r in rows])*100:.0f}%")
    ex: Dict[str, List[Row]] = {}
    for r in rows:
        ex.setdefault(r.exchange, []).append(r)
    print("\n按交易所：")
    for k, v in sorted(ex.items()):
        m = np.mean([r.sm_ret for r in v])
        print(f"  {k:<6} n={len(v):2}  本版均 {m*100:+6.1f}%  正收益 {sum(1 for r in v if r.sm_ret>0)}/{len(v)}")


def main():
    rows = run()
    report(rows)
    # 对比旧版（纯做多/多空）
    from lon_futures_backtest import backtest_product
    lo_all, ls_all = [], []
    for f in sorted(os.listdir(DATA_DIR)):
        if not f.endswith('.parquet'):
            continue
        df = pd.read_parquet(os.path.join(DATA_DIR, f)).rename(columns={'datetime': 'date'})
        if len(df) < 100 or df['volume'].sum() <= 0:
            continue
        _, out = backtest_product(df)
        lo_all.append(out['long_only'][0])
        ls_all.append(out['long_short'][0])
    print(f"\n=== 三版对比(全市场均值) ===")
    print(f"  旧·纯做多(始终在场)    : {np.mean(lo_all)*100:+.1f}%")
    print(f"  旧·多空双向(始终反手)  : {np.mean(ls_all)*100:+.1f}%")
    print(f"  新·止损止盈状态机      : {np.mean([r.sm_ret for r in rows])*100:+.1f}%")


if __name__ == "__main__":
    main()
