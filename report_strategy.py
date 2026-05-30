"""把'均值回归+宏观方向过滤'代表配置做成真实组合资金曲线, 计算完整绩效。

组合口径: 65个品种等权(各 1/N 资金), 逐根K线按持仓盯市, 仓位变动计 2bp/边成本,
按自然日聚合后年化。报告全样本与样本外(全局70%日期切分)两套指标。
"""
import glob
import numpy as np
import pandas as pd
import backtest_qiumo as bt

WARMUP = 1500
Z_IN, W, Z_STOP, MAXH = 2.0, 120, np.inf, 200
COST = bt.COST
files = sorted(glob.glob("data_futures/15min/*.parquet"))


def positions(c, z, mamc):
    """逐bar持仓: 收盘建仓, pos_held[i]=收盘后仓位。均值回归+宏观方向过滤。"""
    n = len(c)
    pos = np.zeros(n)
    p = 0; ei = -1
    for i in range(WARMUP, n):
        zi = z[i]
        if not np.isnan(zi):
            if p > 0 and (zi >= 0 or zi <= -Z_STOP or (i - ei) >= MAXH):
                p = 0
            elif p < 0 and (zi <= 0 or zi >= Z_STOP or (i - ei) >= MAXH):
                p = 0
            if p == 0:
                if zi <= -Z_IN and c[i] > mamc[i]:
                    p, ei = 1, i
                elif zi >= Z_IN and c[i] < mamc[i]:
                    p, ei = -1, i
        pos[i] = p
    return pos


# 逐品种 -> 每日PnL序列
daily = {}            # product -> Series(date -> ret)
trade_rets = []       # 全样本逐笔(整仓)收益, 用于胜率/盈亏比
for f in files:
    df = pd.read_parquet(f).sort_values("datetime").reset_index(drop=True)
    if len(df) < WARMUP + 400:
        continue
    C = df["close"]; ma250 = bt.MA(C, 250); spd = C - ma250
    z = (spd / spd.rolling(W, min_periods=W).std()).to_numpy()
    mamc = bt.MA(C, bt.MACRO).to_numpy()
    c = C.to_numpy(); n = len(c)
    pos = positions(c, z, mamc)
    ret = np.zeros(n)
    ret[1:] = pos[:-1] * (c[1:] / c[:-1] - 1)            # 持仓盯市
    ret -= np.abs(np.diff(pos, prepend=0)) * COST        # 换手成本
    s = pd.Series(ret, index=pd.to_datetime(df["datetime"]))
    daily[f.split("/")[-1][:-8]] = s.groupby(s.index.normalize()).sum()
    # 逐笔(整仓)收益
    p = 0; epx = 0
    for i in range(WARMUP, n):
        if pos[i] != p:
            if p != 0:
                trade_rets.append((c[i] / epx - 1) * p - 2 * COST)
            p = pos[i]; epx = c[i] if p != 0 else epx

port = pd.DataFrame(daily).fillna(0.0).mean(axis=1).sort_index()   # 等权组合日收益

# 全局70%日期切分
dates = port.index
split_date = dates[int(len(dates) * 0.70)]


def metrics(r, label):
    r = r[r.index.notnull()]
    eq = (1 + r).cumprod()
    days = (r.index[-1] - r.index[0]).days or 1
    ann_ret = eq.iloc[-1] ** (365.0 / days) - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe = (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0
    dn = r[r < 0].std()
    sortino = (r.mean() / dn * np.sqrt(252)) if dn > 0 else 0
    mdd = (eq / eq.cummax() - 1).min()
    print(f"[{label}]  天数={days}  总收益={eq.iloc[-1]-1:+.1%}  年化={ann_ret:+.1%}  "
          f"年化波动={ann_vol:.1%}  夏普={sharpe:.2f}  Sortino={sortino:.2f}  "
          f"最大回撤={mdd:.1%}  日胜率={(r>0).mean():.1%}")
    return eq


tr = np.array(trade_rets)
w = tr[tr > 0]; l = tr[tr <= 0]
pf = w.sum() / -l.sum()
print("=" * 96)
print(f"策略: 均值回归 + 宏观方向过滤  (Z_IN={Z_IN}, 窗口W={W}, 无硬止损, 超时{MAXH}根)")
print(f"组合: {len(daily)}品种等权, 成本2bp/边")
print("=" * 96)
print("【逐笔(整仓)统计·全样本】")
print(f"  交易数={len(tr)}  胜率={(tr>0).mean():.1%}  平均盈利={w.mean():+.2%}  "
      f"平均亏损={l.mean():+.2%}  盈亏比={w.mean()/-l.mean():.2f}  PF={pf:.2f}")
print(f"  单笔均收益(净)={tr.mean():+.3%}  最好={tr.max():+.1%}  最差={tr.min():+.1%}")
print("\n【组合资金曲线·年化指标】")
metrics(port, "全样本")
metrics(port[port.index <= split_date], "训练(前70%)")
metrics(port[port.index > split_date], f"样本外(>{split_date.date()})")
# 平均持仓占用(暴露)
expo = np.mean([ (pd.read_parquet(f).shape[0]) for f in files[:0]] or [0])
print("\n注: 组合为1/N恒定等权(多数时间空仓), 故绝对年化偏低; 看夏普/回撤更有意义。")
