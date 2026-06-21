"""
LON 趋势确认策略（LON + LONMA 金叉/死叉 + 零轴）

做多流程：
  1) 零轴下方 LON 金叉 LONMA  -> 进入"关注"(不动手)
  2) LON 上穿零轴(绿柱翻红)    -> 买入
  3) 红柱持续变长              -> 持股不动
  4) 红柱开始缩短              -> 警惕(不动作)
  5) LON 下穿 LONMA 死叉       -> 平仓
做空同理（上下镜像）：
  零轴上方 LON 死叉 LONMA -> 关注；LON 下穿零轴(红柱翻绿) -> 开空；
  LON 上穿 LONMA 金叉 -> 平仓。

相比"抓反转"版，本版是顺势：用金叉确认拐头、零轴确认趋势翻转、死叉离场，
交叉信号天然低频，手续费友好。可在 15 分钟或(合成的)日线上测试。
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.indicators.lon import lon
from lon_futures_backtest import DATA_DIR, MANIFEST, COMMISSION, DEA_PERIOD
from lon_long_short_backtest import _simulate, _metrics, _trade_stats


def build_targets_trend(df: pd.DataFrame) -> np.ndarray:
    """LON+LONMA 趋势确认状态机，返回每根 K 线目标仓位 (+1/-1/0)。"""
    ld = lon(df)
    L = ld['lon'].values
    MA = ld['lonma'].values
    n = len(df)
    warm = max(DEA_PERIOD, 6) + 3

    targets = np.zeros(n)
    pos = 0
    watch = 0  # +1 关注多 / -1 关注空 / 0 无
    for t in range(n):
        if t < 1:
            targets[t] = 0
            continue
        gc = (L[t] > MA[t]) and (L[t - 1] <= MA[t - 1])   # 金叉
        dc = (L[t] < MA[t]) and (L[t - 1] >= MA[t - 1])   # 死叉
        up0 = (L[t] > 0) and (L[t - 1] <= 0)              # 上穿零轴
        dn0 = (L[t] < 0) and (L[t - 1] >= 0)              # 下穿零轴
        lon_t = L[t]

        # 离场（死叉平多 / 金叉平空）
        if pos == 1 and dc:
            pos = 0
        elif pos == -1 and gc:
            pos = 0

        # 进场 / 关注（仅空仓时）
        if pos == 0 and t >= warm:
            if watch == 1 and up0:
                pos = 1
                watch = 0
            elif watch == -1 and dn0:
                pos = -1
                watch = 0
            else:
                if gc and lon_t < 0:        # 零轴下方金叉 -> 关注多
                    watch = 1
                elif dc and lon_t > 0:      # 零轴上方死叉 -> 关注空
                    watch = -1
                elif watch == 1 and dc:     # 关注多途中又死叉 -> 取消关注
                    watch = 0
                elif watch == -1 and gc:    # 关注空途中又金叉 -> 取消关注
                    watch = 0
        targets[t] = pos
    return targets


def _agg(df: pd.DataFrame, k: int) -> pd.DataFrame:
    if k <= 1:
        return df
    g = df.groupby(np.arange(len(df)) // k)
    return pd.DataFrame({
        'date': g['date'].last(), 'open': g['open'].first(), 'high': g['high'].max(),
        'low': g['low'].min(), 'close': g['close'].last(), 'volume': g['volume'].sum(),
    }).reset_index(drop=True)


@dataclass
class Row:
    product: str
    name: str
    exchange: str
    bars: int
    bh: float
    ret: float
    dd: float
    sharpe: float
    trades: int
    win: float
    exposure: float
    vol: float


def run(k: int, ppy: int) -> List[Row]:
    man = pd.read_csv(MANIFEST).set_index('product')
    rows: List[Row] = []
    for f in sorted(os.listdir(DATA_DIR)):
        if not f.endswith('.parquet'):
            continue
        p = f[:-len('.parquet')]
        raw = pd.read_parquet(os.path.join(DATA_DIR, f)).rename(columns={'datetime': 'date'})
        if len(raw) < 100 or raw['volume'].sum() <= 0:
            continue
        df = _agg(raw, k)
        if len(df) < 60:
            continue
        closes = df['close'].values
        tgt = build_targets_trend(df)
        sim = _simulate(tgt, closes, commission=COMMISSION)
        total, dd, sharpe = _metrics(sim['ret'], sim['equity'], ppy)
        trades, win = _trade_stats(tgt, sim['equity'])
        vol = np.diff(np.log(raw['close'].values)).std() * np.sqrt(252 * 16)
        info = man.loc[p] if p in man.index else None
        rows.append(Row(
            product=p, name=str(info['name']) if info is not None else p,
            exchange=str(info['exchange']) if info is not None else '?',
            bars=len(df), bh=float(closes[-1] / closes[0] - 1.0),
            ret=total, dd=dd, sharpe=sharpe, trades=trades, win=win,
            exposure=float((tgt != 0).mean()), vol=vol,
        ))
    return rows


def summarize(tf: str, rows: List[Row]) -> None:
    ret = np.array([r.ret for r in rows]); bh = np.array([r.bh for r in rows])
    print(f"\n=== {tf}：趋势确认版 ===")
    print(f"  全市场均收益 {ret.mean()*100:+.1f}% | 中位 {np.median(ret)*100:+.1f}% | "
          f"正收益 {int((ret>0).sum())}/{len(rows)} | 跑赢B&H {int((ret>bh).sum())}/{len(rows)} | "
          f"平均交易 {np.mean([r.trades for r in rows]):.0f} | 平均在场 {np.mean([r.exposure for r in rows])*100:.0f}% | "
          f"夏普均 {np.mean([r.sharpe for r in rows]):.2f}")
    top = sorted(rows, key=lambda r: r.ret, reverse=True)[:10]
    print("  TOP10:", "  ".join(f"{r.name[:4]}{r.ret*100:+.0f}%" for r in top))
    for lo, hi, lab in [(0, .15, '低波<15%'), (.15, .30, '中波15-30%'), (.30, 9, '高波>30%')]:
        sub = [r for r in rows if lo <= r.vol < hi]
        if sub:
            rr = np.array([r.ret for r in sub])
            print(f"    {lab:<11} n={len(sub):2}  均 {rr.mean()*100:+6.1f}%  正 {int((rr>0).sum())}/{len(sub)}")


def write_md(all_rows: Dict[str, List[Row]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    L = ["# LON 趋势确认策略（LON+LONMA 金叉/死叉 + 零轴）\n\n"]
    L.append("做多：零轴下方金叉→关注；LON 上穿零轴→买入；死叉→平仓。做空镜像。"
             "数据：仓库 `data_futures/15min/`（75 品种），手续费万3。\n\n")
    for tf, rows in all_rows.items():
        ret = np.array([r.ret for r in rows]); bh = np.array([r.bh for r in rows])
        L.append(f"## {tf}\n\n全市场均收益 **{ret.mean()*100:+.1f}%**，中位 {np.median(ret)*100:+.1f}%，"
                 f"正收益 {int((ret>0).sum())}/{len(rows)}，跑赢买入持有 {int((ret>bh).sum())}/{len(rows)}，"
                 f"平均交易 {np.mean([r.trades for r in rows]):.0f} 笔，夏普均 {np.mean([r.sharpe for r in rows]):.2f}。\n\n")
        L.append("| 品种 | 名称 | 所 | 买入持有 | 策略 | 回撤 | 夏普 | 交易 | 胜率 | 波动 |\n")
        L.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for r in sorted(rows, key=lambda r: r.ret, reverse=True):
            L.append(f"| {r.product} | {r.name} | {r.exchange} | {r.bh*100:+.1f}% | {r.ret*100:+.1f}% | "
                     f"{r.dd*100:.1f}% | {r.sharpe:.2f} | {r.trades} | {r.win*100:.0f}% | {r.vol*100:.0f}% |\n")
        L.append("\n")
    open(path, "w", encoding="utf-8").write("".join(L))
    print(f"\n📄 报告已写入: {path}")


def main():
    all_rows = {}
    for tf, (k, ppy) in {'15分钟': (1, 252 * 16), '日线': (16, 252)}.items():
        rows = run(k, ppy)
        all_rows[tf] = rows
        summarize(tf, rows)
    write_md(all_rows, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "results", "lon_trend_report.md"))


if __name__ == "__main__":
    main()
