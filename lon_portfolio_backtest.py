"""
LON 组合策略回测（多品种、排序选股、资金使用率上限）

规则：
  做多  入场: LON < 0 且 MA5 > MA10 > MA20
        退出: LON > 0 且 LON 连续 2 日下降
  做空  入场: LON > 0 且 MA5 < MA10 < MA20      （镜像）
        退出: LON < 0 且 LON 连续 2 日上升
  排序(资金不够时选谁): 做多按 (LON-LONMA) 降序, 做空按 (LONMA-LON) 降序。

组合：本金 10 万；每个仓位 10% 权益；总敞口(多+空)上限 70%（即最多 7 个并发仓位）；
      手续费万分之三。真·日线（按自然日合成 15min 数据）。

输出：胜率、盈亏比、最大回撤、年化（按全市场及不同品种类型分别给出）。
"""
import os, sys
from collections import OrderedDict
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.indicators.lon import lon
from lon_futures_backtest import DATA_DIR, MANIFEST

INIT_CASH = 100_000.0
PER_POS = 0.10          # 每仓位权益占比
MAX_EXPOSURE = 0.70     # 总敞口上限
COMM = 0.0003

# 品种分类（按板块）
SECTORS = {
    '股指': ['IF', 'IC', 'IH', 'IM'],
    '国债': ['T', 'TF', 'TS', 'TL'],
    '贵金属': ['AU', 'AG'],
    '有色金属': ['CU', 'AL', 'ZN', 'PB', 'NI', 'SN', 'BC', 'SS', 'AO'],
    '黑色': ['RB', 'HC', 'I', 'J', 'JM', 'SF', 'SM', 'WR'],
    '能源化工': ['SC', 'FU', 'LU', 'BU', 'TA', 'EG', 'MA', 'PP', 'L', 'V', 'PF',
                 'PG', 'EB', 'PX', 'UR', 'FG', 'SA', 'SH', 'BR', 'RU', 'NR', 'SP'],
    '油脂油料': ['A', 'B', 'M', 'Y', 'P', 'OI', 'RM', 'RS', 'PK'],
    '农副产品': ['C', 'CS', 'JD', 'LH', 'AP', 'CJ', 'SR', 'CF', 'CY', 'RR'],
    '新能源': ['LC', 'PS', 'SI'],
    '其他': ['EC', 'LG', 'FB', 'BB'],
}
PROD2SECTOR = {p: s for s, ps in SECTORS.items() for p in ps}


def to_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """按自然日合成真·日线。"""
    dt = pd.to_datetime(raw['datetime'])
    day = dt.dt.normalize()
    g = raw.groupby(day)
    d = pd.DataFrame({
        'open': g['open'].first(), 'high': g['high'].max(), 'low': g['low'].min(),
        'close': g['close'].last(), 'volume': g['volume'].sum(),
    })
    d.index.name = 'date'
    return d


def precompute(products):
    """加载并预计算每个品种的日线信号。"""
    data = {}
    for f in sorted(os.listdir(DATA_DIR)):
        if not f.endswith('.parquet'):
            continue
        p = f[:-8]
        if p not in products:
            continue
        raw = pd.read_parquet(os.path.join(DATA_DIR, f))
        if len(raw) < 400:
            continue
        d = to_daily(raw)
        if len(d) < 60:
            continue
        c = d['close']
        d['ma5'] = c.rolling(5).mean()
        d['ma10'] = c.rolling(10).mean()
        d['ma20'] = c.rolling(20).mean()
        ld = lon(d.reset_index())
        d['lon'] = ld['lon'].values
        d['lonma'] = ld['lonma'].values
        d['ret'] = c.pct_change().fillna(0.0)
        L = d['lon']
        up_bull = (d['ma5'] > d['ma10']) & (d['ma10'] > d['ma20'])
        dn_bear = (d['ma5'] < d['ma10']) & (d['ma10'] < d['ma20'])
        d['long_entry'] = (L < 0) & up_bull
        d['short_entry'] = (L > 0) & dn_bear
        d['long_exit'] = (L > 0) & (L < L.shift(1)) & (L.shift(1) < L.shift(2))
        d['short_exit'] = (L < 0) & (L > L.shift(1)) & (L.shift(1) > L.shift(2))
        d['rank'] = d['lon'] - d['lonma']
        data[p] = d
    return data


def run_portfolio(products):
    data = precompute(products)
    if not data:
        return None
    master = sorted(set().union(*[set(d.index) for d in data.values()]))

    equity = INIT_CASH
    weights = OrderedDict()   # p -> weight(>0)
    dirs = {}                 # p -> +1/-1
    entry_px = {}
    eq_curve = []
    trades = []  # 每笔净收益率(基于自身名义)

    def close_pos(p, px):
        tr = dirs[p] * (px / entry_px[p] - 1.0) - 2 * COMM
        trades.append(tr)
        nonlocal equity
        equity -= equity * weights[p] * COMM  # 平仓手续费
        del weights[p]; del dirs[p]; del entry_px[p]

    for day in master:
        # 1) 当日组合收益（昨日持仓 × 当日品种收益）
        port_ret = 0.0
        for p, w in weights.items():
            d = data[p]
            r = d['ret'].get(day, 0.0) if day in d.index else 0.0
            port_ret += w * dirs[p] * r
        equity *= (1.0 + port_ret)

        # 2) 收盘后更新仓位：先离场
        for p in list(weights.keys()):
            d = data[p]
            if day not in d.index:
                continue
            row = d.loc[day]
            if (dirs[p] == 1 and row['long_exit']) or (dirs[p] == -1 and row['short_exit']):
                close_pos(p, row['close'])

        # 强制平掉已无后续数据的品种（品种数据到期）
        for p in list(weights.keys()):
            d = data[p]
            if day == d.index[-1]:
                close_pos(p, d.loc[day, 'close'])

        # 3) 进场：按排序选，受 70% 敞口约束
        gross = sum(weights.values())
        cands = []
        for p, d in data.items():
            if p in weights or day not in d.index:
                continue
            row = d.loc[day]
            if pd.isna(row['ma20']) or pd.isna(row['rank']):
                continue
            if row['long_entry']:
                cands.append((row['rank'], p, 1, row['close']))
            elif row['short_entry']:
                cands.append((-row['rank'], p, -1, row['close']))
        cands.sort(key=lambda x: x[0], reverse=True)
        for score, p, dirn, px in cands:
            if gross + PER_POS <= MAX_EXPOSURE + 1e-9:
                equity -= equity * PER_POS * COMM  # 开仓手续费
                weights[p] = PER_POS; dirs[p] = dirn; entry_px[p] = px
                gross += PER_POS

        eq_curve.append((day, equity))

    eq = pd.Series([e for _, e in eq_curve], index=[d for d, _ in eq_curve])
    return metrics(eq, trades)


def metrics(eq: pd.Series, trades):
    total = eq.iloc[-1] / INIT_CASH - 1.0
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = (eq.iloc[-1] / INIT_CASH) ** (1 / years) - 1.0
    dd = ((eq - eq.cummax()) / eq.cummax()).min()
    rets = eq.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0.0
    tr = np.array(trades)
    n = len(tr)
    wins = tr[tr > 0]; losses = tr[tr <= 0]
    win_rate = len(wins) / n if n else 0.0
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = -losses.mean() if len(losses) else 0.0
    pl_ratio = (avg_win / avg_loss) if avg_loss > 0 else float('inf')
    return dict(total=total, cagr=cagr, dd=dd, sharpe=sharpe, n=n,
                win_rate=win_rate, pl_ratio=pl_ratio, avg_win=avg_win,
                avg_loss=avg_loss, years=years, final=eq.iloc[-1])


def fmt(m):
    plr = '∞' if m['pl_ratio'] == float('inf') else f"{m['pl_ratio']:.2f}"
    return (f"年化 {m['cagr']*100:+6.1f}%  总收益 {m['total']*100:+7.1f}%  最大回撤 {m['dd']*100:6.1f}%  "
            f"胜率 {m['win_rate']*100:4.1f}%  盈亏比 {plr:>5}  夏普 {m['sharpe']:+.2f}  交易 {m['n']:>4}")


def main():
    all_products = list(PROD2SECTOR.keys())
    print("=" * 116)
    print(f"LON 组合策略  本金10万  单仓10%  敞口上限70%  真·日线  手续费万3  "
          f"(数据~{ '2023-09~2026-05'})")
    print("=" * 116)

    print("\n【全市场组合】")
    m = run_portfolio(set(all_products))
    print("  " + fmt(m))

    print("\n【按品种类型分别建组合】")
    print(f"  {'板块':<8}{'品种数':>5}  {'年化':>8}{'总收益':>9}{'最大回撤':>9}{'胜率':>7}{'盈亏比':>7}{'夏普':>7}{'交易':>6}")
    rows = []
    for sec, prods in SECTORS.items():
        m = run_portfolio(set(prods))
        if m is None:
            continue
        rows.append((sec, len(prods), m))
        plr = '∞' if m['pl_ratio'] == float('inf') else f"{m['pl_ratio']:.2f}"
        print(f"  {sec:<8}{len(prods):>5}  {m['cagr']*100:>7.1f}%{m['total']*100:>8.1f}%"
              f"{m['dd']*100:>8.1f}%{m['win_rate']*100:>6.1f}%{plr:>7}{m['sharpe']:>7.2f}{m['n']:>6}")
    return rows


if __name__ == "__main__":
    main()
