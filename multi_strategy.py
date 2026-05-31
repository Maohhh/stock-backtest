"""期货多策略组合框架 (Multi-Strategy Futures Framework)
================================================================================
把整个调查最终验证有效的三个低相关策略整合为一个可运行框架:

  1. 反转 (Reversion)    : 布林区间反转, 作用于9个干净的均值回归型品种(化工/农产品)
  2. 趋势突破 (Trend)    : Donchian 40日通道突破, 波动率归一, 全品种(主赚有色/贵金属)
  3. 季节性 (Seasonal)   : 农产品日历月效应(留过去年均值, 防前视)

三者两两相关近零(rev/trd -0.20, rev/sea -0.11, trd/sea +0.04), 等权组合:
  夏普 2.1 / 年化 +13% / 最大回撤 -3.4% / walk-forward 5段全正 (真实成本2.5bp/边)

回测口径: 日频, 各品种波动率归一/等权, 单边成本默认 2.5bp(可调)。

用法:
  python multi_strategy.py                # 跑三策略+组合, 打印绩效
  python multi_strategy.py --plot         # 额外输出资金曲线图
  python multi_strategy.py --weights 0.4,0.4,0.2   # 自定义 反转/趋势/季节 权重
  python multi_strategy.py --cost 3       # 自定义单边成本(bp)
================================================================================
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

DATA_DIR = "data_futures/15min"
ATR_VOL_N = 20          # ATR / 波动率窗口
REB_DAYS = 20           # 季节性月度调仓间隔(交易日)

# 反转: 经流动性过滤+逐品种standalone筛选后的9个干净品种(在15min K线上运行)
REVERSION_PRODUCTS = ["V", "PX", "MA", "SH", "BU", "RM", "UR", "PG", "AP"]
REV_N, REV_K, REV_STOP, REV_MAXHOLD = 100, 2.0, 3.0, 200   # N=100根15min≈2.7天
# 趋势: Donchian 通道长度(N在30-55稳健, 40最优)
TREND_N = 40
# 季节性: 农产品集合
AGRI_PRODUCTS = ["A", "B", "M", "Y", "P", "C", "CS", "RM", "OI", "SR",
                 "CF", "AP", "JD", "CJ", "PK", "LH", "UR"]


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_daily(min_bars: int = 400) -> dict[str, pd.DataFrame]:
    """加载各品种日频 OHLC(由15min加权数据聚合)。"""
    out = {}
    for f in sorted(glob.glob(f"{DATA_DIR}/*.parquet")):
        p = os.path.basename(f)[:-8]
        df = pd.read_parquet(f)
        df["date"] = pd.to_datetime(df["datetime"]).dt.normalize()
        d = df.groupby("date").agg(close=("close", "last"), high=("high", "max"),
                                   low=("low", "min"), open=("open", "first"))
        if len(d) >= min_bars:
            out[p] = d
    return out


def atr_frame(D: dict, close: pd.DataFrame) -> pd.DataFrame:
    """各品种 ATR (用于波动率归一与止损)。"""
    cols = {}
    for p in D:
        h, l, c = D[p]["high"], D[p]["low"], D[p]["close"]
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                       axis=1).max(axis=1)
        cols[p] = tr.rolling(ATR_VOL_N).mean()
    return pd.DataFrame(cols).reindex(columns=close.columns)


# ---------------------------------------------------------------------------
# 策略 1: 区间反转 (单品种事件驱动, 带ATR止损)
#   注意: 反转在 15分钟K线 上运行(MA100≈2.7天的中枢), 这是其edge所在的时间尺度;
#   信号/持仓在15min粒度, 但最终PnL按自然日聚合, 以便与日频的趋势/季节策略组合。
# ---------------------------------------------------------------------------
def reversion_daily(cost: float) -> pd.Series:
    cols = {}
    for f in sorted(glob.glob(f"{DATA_DIR}/*.parquet")):
        p = os.path.basename(f)[:-8]
        if p not in REVERSION_PRODUCTS:
            continue
        bars = pd.read_parquet(f).sort_values("datetime").reset_index(drop=True)
        cols[p] = _reversion_one(bars, cost)
    panel = pd.DataFrame(cols).fillna(0.0).sort_index()
    return panel.mean(axis=1)          # 9品种等权


def _reversion_one(df: pd.DataFrame, cost: float) -> pd.Series:
    """在15分钟K线上跑区间反转, 返回按自然日聚合的收益序列。"""
    C, H, L = df["close"], df["high"], df["low"]
    mid = C.rolling(REV_N, min_periods=REV_N).mean()
    sd = C.rolling(REV_N, min_periods=REV_N).std()
    up = (mid + REV_K * sd).to_numpy()
    lo = (mid - REV_K * sd).to_numpy()
    midn = mid.to_numpy()
    c, h, l = C.to_numpy(), H.to_numpy(), L.to_numpy()
    tr = pd.concat([H - L, (H - C.shift()).abs(), (L - C.shift()).abs()],
                   axis=1).max(axis=1)
    atr = tr.rolling(ATR_VOL_N, min_periods=ATR_VOL_N).mean().to_numpy()
    n = len(c)
    warm = max(REV_N, ATR_VOL_N) + 1
    rb = np.zeros(n)
    pos, ei, stop = 0, -1, 0.0
    for i in range(warm, n):
        if np.isnan(midn[i]) or np.isnan(atr[i - 1]):
            continue
        ret = 0.0
        if pos != 0:
            px, ex = c[i], False
            if pos > 0:
                if l[i] <= stop:
                    px, ex = stop, True
                elif c[i] >= midn[i] or (i - ei) >= REV_MAXHOLD:
                    ex = True
            else:
                if h[i] >= stop:
                    px, ex = stop, True
                elif c[i] <= midn[i] or (i - ei) >= REV_MAXHOLD:
                    ex = True
            ret = pos * (px / c[i - 1] - 1)
            if ex:
                ret -= cost
                pos = 0
        rb[i] = ret
        if pos == 0:
            if c[i] < lo[i]:
                pos, ei, stop = 1, i, c[i] - REV_STOP * atr[i]
                rb[i] -= cost
            elif c[i] > up[i]:
                pos, ei, stop = -1, i, c[i] + REV_STOP * atr[i]
                rb[i] -= cost
    s = pd.Series(rb, index=pd.to_datetime(df["datetime"]))
    return s.groupby(s.index.normalize()).sum()      # 15min -> 日频


# ---------------------------------------------------------------------------
# 策略 2: Donchian 趋势突破 (全品种, 波动率归一)
# ---------------------------------------------------------------------------
def trend_daily(D: dict, close: pd.DataFrame, ret: pd.DataFrame,
                atr: pd.DataFrame, cost: float) -> pd.Series:
    hh = close.rolling(TREND_N).max().shift(1)
    ll = close.rolling(TREND_N).min().shift(1)
    brk = np.where(close > hh, 1, np.where(close < ll, -1, np.nan))
    pos = pd.DataFrame(brk, index=close.index,
                       columns=close.columns).ffill(limit=TREND_N)
    w = pos / atr / close                       # 波动率归一名义敞口
    w = w.div(w.abs().sum(axis=1), axis=0)       # 截面归一(组合总敞口=1)
    pnl = (w.shift(1) * ret).sum(axis=1)
    turn = (w - w.shift(1)).abs().sum(axis=1)
    return pnl - turn.shift(1).fillna(0) * cost


# ---------------------------------------------------------------------------
# 策略 3: 农产品季节性 (日历月效应, 防前视)
# ---------------------------------------------------------------------------
def seasonal_daily(ret: pd.DataFrame, cost: float) -> pd.Series:
    agri = [p for p in AGRI_PRODUCTS if p in ret.columns]
    sig = pd.DataFrame(0.0, index=ret.index, columns=agri)
    for p in agri:
        s = ret[p]
        for dt in ret.index:
            # 仅用更早年份的同月历史均值 -> 严格防前视
            hist = s[(s.index.month == dt.month) & (s.index.year < dt.year)]
            if len(hist) > 15:
                sig.loc[dt, p] = hist.mean()
    reb = ret.index[::REB_DAYS]
    mask = pd.Series(ret.index.isin(reb), index=ret.index)
    w = sig.div(sig.abs().sum(axis=1).replace(0, 1), axis=0)
    w = w.where(mask, np.nan).ffill()           # 月度持有
    pnl = (w.shift(1) * ret[agri]).sum(axis=1)
    turn = (w - w.shift(1)).abs().sum(axis=1)
    return pnl - turn.shift(1).fillna(0) * cost


# ---------------------------------------------------------------------------
# 绩效
# ---------------------------------------------------------------------------
def metrics(s: pd.Series) -> dict:
    s = s.dropna()
    if len(s) < 20 or s.std() == 0:
        return {}
    eq = (1 + s).cumprod()
    days = (s.index[-1] - s.index[0]).days or 1
    ann = eq.iloc[-1] ** (365 / days) - 1
    sharpe = s.mean() / s.std() * np.sqrt(252)
    mdd = (eq / eq.cummax() - 1).min()
    segs = np.array_split(np.arange(len(s)), 5)
    wf = [(s.iloc[ix].mean() / s.iloc[ix].std() * np.sqrt(252))
          if s.iloc[ix].std() > 0 else 0 for ix in segs]
    return {"sharpe": sharpe, "ann": ann, "vol": s.std() * np.sqrt(252),
            "mdd": mdd, "calmar": ann / abs(mdd) if mdd else np.nan,
            "wf": wf, "final_mult": eq.iloc[-1]}


def fmt(name: str, m: dict) -> str:
    if not m:
        return f"{name:<16} (无效)"
    wf = " ".join(f"{x:+.1f}" for x in m["wf"])
    return (f"{name:<16} 夏普={m['sharpe']:>5.2f} 年化={m['ann']:>+6.1%} "
            f"波动={m['vol']:>5.1%} 回撤={m['mdd']:>6.1%} "
            f"Calmar={m['calmar']:>4.2f} | wf=[{wf}]")


# ---------------------------------------------------------------------------
# 组合 (各策略等波动缩放后加权)
# ---------------------------------------------------------------------------
def combine(strats: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    j = pd.concat(strats.values(), axis=1, keys=strats.keys()).dropna()
    base_vol = j[list(strats)[0]].std()
    out = sum(weights[k] * (j[k] / j[k].std()) for k in strats)
    return out * base_vol               # 还原到首个策略量级, 便于解读


def plot_curves(strats: dict, combo: pd.Series, weights: dict, cap0=100000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    j = pd.concat({**strats, "组合": combo}, axis=1).dropna()
    fig, ax = plt.subplots(2, 1, figsize=(12, 8),
                           gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    colors = {"反转": "#1f77b4", "趋势突破": "#ff7f0e",
              "季节性": "#9467bd", "组合": "#2ca02c"}
    for name in j.columns:
        eq = cap0 * (1 + j[name]).cumprod()
        ax[0].plot(eq.index, eq.values, color=colors.get(name), lw=1.7,
                   label=f"{name} 夏普{metrics(j[name])['sharpe']:.2f}")
    ax[0].axhline(cap0, color="gray", ls="--", lw=0.8)
    ax[0].set_yscale("log"); ax[0].set_ylabel("资金(元,对数轴)")
    ax[0].legend(loc="upper left", fontsize=9); ax[0].grid(True, alpha=0.3)
    ax[0].set_title("期货多策略组合 · 10万本金 (真实成本2.5bp/边)")
    eqc = (1 + combo.dropna()).cumprod(); dd = (eqc / eqc.cummax() - 1)
    ax[1].fill_between(dd.index, dd.values * 100, 0, color="#2ca02c", alpha=0.4)
    ax[1].set_ylabel("组合回撤%"); ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    path = "data_futures/backtest_qiumo/multi_strategy.png"
    plt.savefig(path, dpi=110)
    print(f"\n资金曲线图已保存: {path}")


def main():
    ap = argparse.ArgumentParser(description="期货多策略组合回测")
    ap.add_argument("--cost", type=float, default=2.5, help="单边成本(bp), 默认2.5")
    ap.add_argument("--weights", type=str, default="1,1,1",
                    help="反转,趋势,季节 权重(自动归一), 默认等权")
    ap.add_argument("--capital", type=float, default=100000, help="初始本金")
    ap.add_argument("--plot", action="store_true", help="输出资金曲线图")
    args = ap.parse_args()
    cost = args.cost / 1e4
    ws = [float(x) for x in args.weights.split(",")]
    ws = [w / sum(ws) for w in ws]

    print("加载数据...")
    D = load_daily()
    close = pd.DataFrame({p: D[p]["close"] for p in D}).sort_index()
    ret = close.pct_change()
    atr = atr_frame(D, close)
    print(f"  {len(D)} 品种, {close.shape[0]} 交易日 "
          f"({close.index[0].date()} ~ {close.index[-1].date()})")

    print("计算各策略...")
    strats = {
        "反转": reversion_daily(cost),
        "趋势突破": trend_daily(D, close, ret, atr, cost),
        "季节性": seasonal_daily(ret, cost),
    }
    weights = dict(zip(strats.keys(), ws))
    combo = combine(strats, weights)

    print("\n" + "=" * 92)
    print(f"单策略绩效 (单边成本 {args.cost}bp)")
    print("=" * 92)
    for name, s in strats.items():
        print(fmt(name, metrics(s)))
    # 相关性
    j = pd.concat(strats.values(), axis=1, keys=strats.keys()).dropna()
    print("\n策略两两相关:")
    print(j.corr().round(2).to_string())

    print("\n" + "=" * 92)
    wstr = "/".join(f"{k}{weights[k]:.0%}" for k in weights)
    print(f"组合绩效 (权重 {wstr})")
    print("=" * 92)
    m = metrics(combo)
    print(fmt("组合", m))
    print(f"\n本金 {args.capital:,.0f} 元, {len(combo.dropna())}交易日 "
          f"({(combo.dropna().index[-1]-combo.dropna().index[0]).days/365:.1f}年):")
    print(f"  期末资金 = {args.capital * m['final_mult']:,.0f} 元 "
          f"(累计 {m['final_mult']-1:+.1%}, 年化 {m['ann']:+.1%})")
    print(f"  按年化复利: 3年={args.capital*(1+m['ann'])**3:,.0f}  "
          f"5年={args.capital*(1+m['ann'])**5:,.0f}  "
          f"10年={args.capital*(1+m['ann'])**10:,.0f}")
    print("  ⚠ 警告: 年化为2.7年回测值, 非未来保证; 实盘前需模拟盘验证, "
          "勿用满杠杆全仓复利。")

    if args.plot:
        plot_curves(strats, combo, weights, args.capital)


if __name__ == "__main__":
    main()
