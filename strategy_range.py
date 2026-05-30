"""分型专配 · CS型(低波均值回归)区间反转策略 + walk-forward。

核心问题: 把策略只用在结构同质的品种上, 能否比"全品种一套"更稳(治非平稳)?

指标: 布林式区间反转
  mid=SMA(N), band=mid±K*std(N)
  触下轨(C<lower)做多 / 触上轨(C>upper)做空
  回到中枢(C穿回mid)止盈; ATR硬止损(峰度极高必须有); 超时离场
对比三组(同一套指标/参数):
  REVERT  = VR40 最低 1/3 (最均值回归)
  TREND   = VR40 最高 1/3 (最趋势/动量)
  CSTYPE  = 低波(vol<中位) 且 VR40<0.9  (玉米淀粉这一族)
  ALL     = 全品种
walk-forward: 时间均分5段各算夏普, 看最差段(min)与是否全正。
口径: 等额, 成本2bp/边。
"""
import glob
import numpy as np
import pandas as pd

ATR_N = 100
COST = 0.0002
MAXH = 200
files = sorted(glob.glob("data_futures/15min/*.parquet"))
prof = pd.read_csv("data_futures/backtest_qiumo/structure_profile.csv").set_index("prod")
prof = prof[~prof.index.duplicated()]


def make_groups():
    vr = prof["VR40"].dropna()
    q1, q2 = vr.quantile(1 / 3), vr.quantile(2 / 3)
    revert = vr[vr <= q1].index.tolist()
    trend = vr[vr >= q2].index.tolist()
    vmed = prof["vol_ann"].median()
    cstype = prof[(prof.vol_ann < vmed) & (prof.VR40 < 0.9)].index.tolist()
    return {"REVERT(最回归1/3)": revert, "TREND(最趋势1/3)": trend,
            "CSTYPE(低波回归)": cstype, "ALL(全品种)": list(prof.index)}


def load(prods):
    out = {}
    for f in files:
        p = f.split("/")[-1][:-8]
        if p not in prods:
            continue
        df = pd.read_parquet(f).sort_values("datetime").reset_index(drop=True)
        out[p] = df
    return out


DATA = {f.split("/")[-1][:-8]: pd.read_parquet(f).sort_values("datetime").reset_index(drop=True)
        for f in files}


def sim_daily(df, N, K, stop_atr):
    """区间反转, 返回 按日聚合的逐bar收益(等额单位)。"""
    C = df["close"]; H = df["high"]; L = df["low"]
    mid = C.rolling(N, min_periods=N).mean()
    sd = C.rolling(N, min_periods=N).std()
    up = (mid + K * sd).to_numpy(); lo = (mid - K * sd).to_numpy()
    midn = mid.to_numpy(); c = C.to_numpy(); h = H.to_numpy(); l = L.to_numpy()
    tr = pd.concat([H - L, (H - C.shift()).abs(), (L - C.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(ATR_N, min_periods=ATR_N).mean().to_numpy()
    n = len(c); warm = max(N, ATR_N) + 1
    rb = np.zeros(n); p = 0; ei = -1; estop = 0.0
    for i in range(warm, n):
        if np.isnan(midn[i]) or np.isnan(atr[i - 1]):
            continue
        ret = 0.0
        if p != 0:
            px = c[i]; ex = False
            if p > 0:
                if l[i] <= estop:
                    px = estop; ex = True
                elif c[i] >= midn[i] or (i - ei) >= MAXH:
                    ex = True
            else:
                if h[i] >= estop:
                    px = estop; ex = True
                elif c[i] <= midn[i] or (i - ei) >= MAXH:
                    ex = True
            ret = p * (px / c[i - 1] - 1)
            if ex:
                ret -= COST; p = 0
        rb[i] = ret
        if p == 0:
            if c[i] < lo[i]:
                p = 1; ei = i; estop = c[i] - stop_atr * atr[i]; rb[i] -= COST
            elif c[i] > up[i]:
                p = -1; ei = i; estop = c[i] + stop_atr * atr[i]; rb[i] -= COST
    s = pd.Series(rb, index=pd.to_datetime(df["datetime"]))
    return s.groupby(s.index.normalize()).sum()


def port_daily(prods, N, K, stop_atr):
    cols = {p: sim_daily(DATA[p], N, K, stop_atr) for p in prods if p in DATA}
    return pd.DataFrame(cols).fillna(0.0).mean(axis=1).sort_index()


def wf(daily, k=5):
    daily = daily.dropna()
    segs = np.array_split(np.arange(len(daily)), k)
    out = []
    for idx in segs:
        r = daily.iloc[idx]
        out.append(r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0)
    full = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    return full, out


if __name__ == "__main__":
    groups = make_groups()
    print("分组规模:", {k: len(v) for k, v in groups.items()})
    N, K, STOP = 100, 2.0, 3.0
    print(f"\n默认配置: 布林N={N} K={K} ATR止损={STOP}  (同一套指标用于各组)")
    print("=" * 92)
    print(f"{'组':<20} | {'全期夏普':>7} | {'walk-forward 5段夏普':>34} | {'最差段':>6}")
    print("=" * 92)
    for gname, prods in groups.items():
        full, segs = wf(port_daily(prods, N, K, STOP))
        segstr = " ".join(f"{s:>5.2f}" for s in segs)
        print(f"{gname:<20} | {full:>7.2f} | {segstr:>34} | {min(segs):>6.2f}")

    print("=" * 92)
    print("CSTYPE 组上 参数稳健性 (N×K, 看全期夏普/最差段夏普)")
    print("=" * 92)
    cs = groups["CSTYPE(低波回归)"]
    for N in [50, 100, 200]:
        for K in [1.5, 2.0, 2.5]:
            full, segs = wf(port_daily(cs, N, K, 3.0))
            print(f"  N={N:<3} K={K} : 全期={full:>5.2f}  最差段={min(segs):>5.2f}  "
                  f"段=[{' '.join(f'{s:+.1f}' for s in segs)}]")
