"""均值回归策略 · 改进方向全量回测(带样本外护栏)。

可配置因子:
  sizing   : 'equal'(等额名义) / 'invvol'(按ATR逆波动定仓, 每笔风险≈恒定)
  stop_k   : ATR硬止损倍数(None=无止损)
  confirm  : 回归确认入场(z仍在极端区但开始拐头才进, 不接飞刀)
  macro    : 宏观方向过滤(只在C>MA(MACRO_L)做多/下方做空)
  blacklist: 剔除高波/低流动品种

研究: A)定仓  B)ATR止损  C)确认入场  D)黑名单  E)组合最优 + walk-forward分段稳定性
口径: bars>=WARMUP, 成本2bp/边, 全局70%日期切训练/样本外。
"""
import glob
import numpy as np
import pandas as pd
import backtest_qiumo as bt

WARMUP = 1500
MACRO_L = 1500
W = 120
Z_IN = 2.0
MAXH = 200
ATR_N = 100
COST = bt.COST
TARGET = 0.006          # invvol目标: 每笔ATR%≈0.6%对应size=1
files = sorted(glob.glob("data_futures/15min/*.parquet"))


def load():
    P = []
    for f in files:
        df = pd.read_parquet(f).sort_values("datetime").reset_index(drop=True)
        if len(df) < WARMUP + 400:
            continue
        C = df["close"]; H = df["high"]; L = df["low"]
        ma250 = bt.MA(C, 250); spd = C - ma250
        z = (spd / spd.rolling(W, min_periods=W).std()).to_numpy()
        tr = pd.concat([H - L, (H - C.shift()).abs(), (L - C.shift()).abs()],
                       axis=1).max(axis=1)
        atr = tr.rolling(ATR_N, min_periods=ATR_N).mean().to_numpy()
        P.append({"prod": f.split("/")[-1][:-8], "dt": pd.to_datetime(df["datetime"]),
                  "c": C.to_numpy(), "h": H.to_numpy(), "l": L.to_numpy(),
                  "z": z, "mamc": bt.MA(C, MACRO_L).to_numpy(), "atr": atr,
                  "atr_pct": (atr / C.to_numpy())})
    return P


def sim(d, *, sizing="equal", stop_k=None, confirm=False, macro=True):
    """返回逐bar已实现收益(unit, 含正确出场价与成本) 与 逐笔(整仓)收益列表。"""
    c, h, l, z, mamc, atr, atrp = (d["c"], d["h"], d["l"], d["z"], d["mamc"],
                                   d["atr"], d["atr_pct"])
    n = len(c)
    rb = np.zeros(n)              # 逐bar unit收益(已计成本)
    sz = np.zeros(n)             # 该bar仓位规模(用于组合)
    trades = []
    p = 0; ei = -1; estop = 0.0; epx = 0.0; size = 0.0
    for i in range(WARMUP, n):
        zi = z[i]
        ret = 0.0
        if p != 0:
            exited = False; px_end = c[i]
            if p > 0:
                if stop_k and not np.isnan(atr[i - 1]) and l[i] <= estop:
                    px_end = estop; exited = True
                elif zi >= 0 or (i - ei) >= MAXH:
                    exited = True
            else:
                if stop_k and not np.isnan(atr[i - 1]) and h[i] >= estop:
                    px_end = estop; exited = True
                elif zi <= 0 or (i - ei) >= MAXH:
                    exited = True
            ret = p * (px_end / c[i - 1] - 1)
            if exited:
                ret -= COST                                  # 出场成本
                trades.append(p * (px_end / epx - 1) - 2 * COST)
                p = 0
        rb[i] = ret * size
        sz[i] = size if p != 0 else 0.0
        if p == 0 and not np.isnan(zi):
            long_sig = zi <= -Z_IN and (not confirm or zi > z[i - 1])
            short_sig = zi >= Z_IN and (not confirm or zi < z[i - 1])
            if macro:
                long_sig = long_sig and c[i] > mamc[i]
                short_sig = short_sig and c[i] < mamc[i]
            if long_sig or short_sig:
                p = 1 if long_sig else -1
                epx = c[i]; ei = i
                if sizing == "invvol" and atrp[i] > 0:
                    size = float(np.clip(TARGET / atrp[i], 0.2, 5.0))
                else:
                    size = 1.0
                estop = (c[i] - stop_k * atr[i]) if (stop_k and p > 0) else \
                        (c[i] + stop_k * atr[i]) if stop_k else 0.0
                rb[i] -= COST * size                          # 进场成本
                sz[i] = size
    return rb, sz, trades


def portfolio(P, split_date, **cfg):
    cols = {}; all_tr = []
    for d in P:
        rb, sz, tr = sim(d, **cfg)
        all_tr += tr
        s = pd.Series(rb, index=d["dt"])
        cols[d["prod"]] = s.groupby(s.index.normalize()).sum()
    daily = pd.DataFrame(cols).fillna(0.0).mean(axis=1).sort_index()
    return daily, np.array(all_tr)


def stat(daily, tr, split_date):
    def met(r):
        r = r.dropna()
        if len(r) < 5 or r.std() == 0:
            return dict(sharpe=0, ann=0, vol=0, mdd=0)
        eq = (1 + r).cumprod()
        days = (r.index[-1] - r.index[0]).days or 1
        return dict(sharpe=r.mean() / r.std() * np.sqrt(252),
                    ann=eq.iloc[-1] ** (365 / days) - 1,
                    vol=r.std() * np.sqrt(252),
                    mdd=(eq / eq.cummax() - 1).min())
    tr_d = daily[daily.index <= split_date]; te_d = daily[daily.index > split_date]
    m_tr = met(tr_d); m_te = met(te_d)
    if len(tr):
        w = tr[tr > 0]; ll = tr[tr <= 0]
        pf = w.sum() / -ll.sum() if ll.sum() < 0 else np.inf
        payoff = w.mean() / -ll.mean() if len(ll) else np.inf
        win = (tr > 0).mean(); worst = tr.min()
    else:
        pf = payoff = win = worst = 0
    return m_tr, m_te, dict(n=len(tr), win=win, payoff=payoff, pf=pf, worst=worst)


def line(label, m_tr, m_te, t):
    print(f"{label:<26} | OOS夏普={m_te['sharpe']:>5.2f} OOS年化={m_te['ann']:>+6.1%} "
          f"OOS回撤={m_te['mdd']:>6.1%} | 训练夏普={m_tr['sharpe']:>5.2f} | "
          f"笔={t['n']:>4} 胜={t['win']:.0%} 盈亏比={t['payoff']:.2f} PF={t['pf']:.2f} "
          f"最差={t['worst']:>6.1%}")


if __name__ == "__main__":
    P = load()
    alld = sorted(set().union(*[set(d["dt"].dt.normalize()) for d in P]))
    split_date = alld[int(len(alld) * 0.70)]
    print(f"品种={len(P)}  训练/样本外切分日={split_date.date()}")
    base = dict(sizing="equal", stop_k=None, confirm=False, macro=True)

    def run(label, **over):
        cfg = {**base, **over}
        daily, tr = portfolio(P, split_date, **cfg)
        m_tr, m_te, t = stat(daily, tr, split_date)
        line(label, m_tr, m_te, t)
        return m_te["sharpe"]

    print("=" * 118); print("基准(当前代表配置)"); print("=" * 118)
    run("base: 等额/无止损/无确认")
    print("=" * 118); print("A) 仓位规模"); print("=" * 118)
    run("A invvol(逆波动定仓)", sizing="invvol")
    print("=" * 118); print("B) ATR硬止损 (在invvol基础上)"); print("=" * 118)
    for k in [1.5, 2.5, 4.0]:
        run(f"B invvol+stop{k}ATR", sizing="invvol", stop_k=k)
    print("=" * 118); print("C) 回归确认入场"); print("=" * 118)
    run("C invvol+确认", sizing="invvol", confirm=True)
    run("C invvol+stop2.5+确认", sizing="invvol", stop_k=2.5, confirm=True)
    print("=" * 118); print("D) 宏观过滤开关 (对照)"); print("=" * 118)
    run("D invvol+无宏观", sizing="invvol", macro=False)
    print("=" * 118); print("E) 组合最优候选"); print("=" * 118)
    run("E invvol+stop2.5+确认+宏观", sizing="invvol", stop_k=2.5, confirm=True, macro=True)
    run("E invvol+stop4+确认+宏观", sizing="invvol", stop_k=4.0, confirm=True, macro=True)

    print("=" * 118); print("F) 隔离: 等额 vs 逆波动 × 止损倍数 (均无确认)"); print("=" * 118)
    for k in [3.0, 4.0, 5.0, 6.0]:
        run(f"F 等额+stop{k}", sizing="equal", stop_k=k)
    for k in [4.0, 5.0, 6.0]:
        run(f"F invvol+stop{k}", sizing="invvol", stop_k=k)

    print("=" * 118); print("G) 最优配置 walk-forward 时间分段稳定性"); print("=" * 118)
    best_cfg = dict(sizing="equal", stop_k=5.0, confirm=False, macro=True)
    daily, tr = portfolio(P, split_date, **best_cfg)
    daily = daily.dropna()
    segs = np.array_split(np.arange(len(daily)), 5)
    print("配置: 等额 + stop5ATR + 宏观  —— 把时间均分5段各算夏普:")
    for j, idx in enumerate(segs):
        r = daily.iloc[idx]
        sh = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
        eq = (1 + r).cumprod()
        print(f"  段{j+1} {r.index[0].date()}~{r.index[-1].date()}: "
              f"夏普={sh:>5.2f} 收益={eq.iloc[-1]-1:>+6.1%} 日数={len(r)}")
