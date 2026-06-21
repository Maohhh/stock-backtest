"""
两个 LON 趋势版对比：

版本A（上一版·趋势确认）：
  0轴下方金叉->关注；LON 上穿零轴(绿翻红)->买入；LON 下穿 LONMA 死叉->平仓。(做空镜像)

版本B（本版）：
  0轴下方金叉->直接买入；0轴上方红柱连续2日缩短->平仓。(做空镜像)
  即：进场更早(金叉即买,不等穿零轴)，离场更早(红柱2日缩短,不等死叉)。

同条件对比：国内期货 75 品种，15分钟与(合成)日线，手续费万3。
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.indicators.lon import lon
from lon_futures_backtest import DATA_DIR, MANIFEST, COMMISSION, DEA_PERIOD
from lon_long_short_backtest import _simulate, _metrics, _trade_stats
from lon_trend_backtest import build_targets_trend, _agg   # 版本A


def build_targets_trend2(df: pd.DataFrame) -> np.ndarray:
    """版本B：金叉(零轴下方)即买入；零轴上方红柱连续2日缩短平仓。做空镜像。"""
    ld = lon(df)
    L = ld['lon'].values
    MA = ld['lonma'].values
    length = np.abs(L)
    n = len(df)
    warm = max(DEA_PERIOD, 6) + 3
    targets = np.zeros(n)
    pos = 0
    for t in range(n):
        if t < 2:
            targets[t] = 0
            continue
        gc = (L[t] > MA[t]) and (L[t - 1] <= MA[t - 1])
        dc = (L[t] < MA[t]) and (L[t - 1] >= MA[t - 1])
        shrink2 = (length[t] < length[t - 1]) and (length[t - 1] < length[t - 2])
        lon_t = L[t]
        # 离场
        if pos == 1 and lon_t > 0 and shrink2:      # 多：零轴上方红柱连续2日缩短
            pos = 0
        elif pos == -1 and lon_t < 0 and shrink2:   # 空：零轴下方绿柱连续2日缩短
            pos = 0
        # 进场
        if pos == 0 and t >= warm:
            if gc and lon_t < 0:        # 零轴下方金叉 -> 买入
                pos = 1
            elif dc and lon_t > 0:      # 零轴上方死叉 -> 做空
                pos = -1
        targets[t] = pos
    return targets


def evalu(tgt, closes, ppy):
    sim = _simulate(tgt, closes, commission=COMMISSION)
    total, dd, sharpe = _metrics(sim['ret'], sim['equity'], ppy)
    trades, win = _trade_stats(tgt, sim['equity'])
    return dict(ret=total, dd=dd, sharpe=sharpe, trades=trades, win=win,
                expo=float((tgt != 0).mean()))


def run(k, ppy):
    man = pd.read_csv(MANIFEST).set_index('product')
    rows = []
    for f in sorted(os.listdir(DATA_DIR)):
        if not f.endswith('.parquet'):
            continue
        p = f[:-8]
        raw = pd.read_parquet(os.path.join(DATA_DIR, f)).rename(columns={'datetime': 'date'})
        if len(raw) < 100 or raw['volume'].sum() <= 0:
            continue
        df = _agg(raw, k)
        if len(df) < 60:
            continue
        c = df['close'].values
        A = evalu(build_targets_trend(df), c, ppy)
        B = evalu(build_targets_trend2(df), c, ppy)
        vol = np.diff(np.log(raw['close'].values)).std() * np.sqrt(252 * 16)
        nm = str(man.loc[p, 'name']) if p in man.index else p
        bh = float(c[-1] / c[0] - 1.0)
        rows.append(dict(p=p, nm=nm, bh=bh, vol=vol, A=A, B=B))
    return rows


def summ(tag, rows):
    A = np.array([r['A']['ret'] for r in rows]); B = np.array([r['B']['ret'] for r in rows])
    bh = np.array([r['bh'] for r in rows])
    print(f"\n=== {tag} ===")
    for lab, X, tr in [('版本A(死叉离场)', A, [r['A']['trades'] for r in rows]),
                       ('版本B(红柱2日缩短离场)', B, [r['B']['trades'] for r in rows])]:
        print(f"  {lab:<22} 均{X.mean()*100:+6.1f}%  中位{np.median(X)*100:+6.1f}%  "
              f"正收益{int((X>0).sum()):2}/{len(rows)}  跑赢B&H{int((X>bh).sum()):2}/{len(rows)}  "
              f"平均交易{np.mean(tr):4.0f}")
    print(f"  B优于A的品种数: {int((B>A).sum())}/{len(rows)}")
    for lo, hi, lb in [(0, .15, '低波'), (.15, .30, '中波'), (.30, 9, '高波')]:
        sub = [r for r in rows if lo <= r['vol'] < hi]
        if sub:
            a = np.mean([r['A']['ret'] for r in sub]); b = np.mean([r['B']['ret'] for r in sub])
            print(f"    {lb}({len(sub):2}): A {a*100:+6.1f}%  B {b*100:+6.1f}%")


def main():
    for tag, (k, ppy) in {'15分钟': (1, 252 * 16), '日线': (16, 252)}.items():
        summ(tag, run(k, ppy))


if __name__ == "__main__":
    main()
