"""对照实验: 验证「病根在入场」的两个改法。

变体(均在 bars>=1500 的同一批K线上, 与原版口径一致):
  V0  原版求魔(带宏观门)            —— 追突破入场(基准, 已知无edge)
  V1  均值回归: 价格极端偏离MA250反向进场(z<=-Z 做多 / z>=+Z 做空)
  V2  动量去追高: 上升趋势中回踩重拾MA20即做多(不要求突破前高), 且让赢家跑

输出:
  A) 各变体入场前向收益 vs 同向基准 (毛, 不经出场) —— 超额>0 才有edge
  B) 各变体实现绩效(配各自合适的出场): 胜率/盈亏比/PF/等权合计
"""
import glob
import numpy as np
import pandas as pd
import backtest_qiumo as bt

WARMUP = 1500
Z_IN = 2.0          # 进场极端阈值(z)
Z_STOP = 3.5        # 反向继续扩大止损
MAXH = 200
COST = bt.COST
H = [10, 20, 40, 80, 160]
files = sorted(glob.glob("data_futures/15min/*.parquet"))


def prep(df):
    """计算各变体所需的基础列与入场信号。"""
    ind = bt.compute_indicators(df, use_macro=True)
    C = ind["close"]
    ma20 = bt.MA(C, 20)
    ma250 = ind["ma250"]
    spd = C - ma250
    z = spd / spd.rolling(250, min_periods=250).std()

    # V1 均值回归: 首次进入极端区
    v1_long = (z <= -Z_IN) & (z.shift(1) > -Z_IN)
    v1_short = (z >= Z_IN) & (z.shift(1) < Z_IN)

    # V2 动量去追高: 趋势向上(C>MA250且MA250上行)中, 回踩后重拾MA20
    upt = (ma250 - ma250.shift(bt.SLP_N)) / C > bt.SLP_E
    dnt = (ma250 - ma250.shift(bt.SLP_N)) / C < -bt.SLP_E
    regime_up = (C > ma250) & upt
    regime_dn = (C < ma250) & dnt
    v2_long = bt.CROSS(C, ma20) & regime_up
    v2_short = bt.CROSS(ma20, C) & regime_dn

    ind["z"] = z.to_numpy()
    ind["v1_long"] = v1_long.fillna(False).to_numpy()
    ind["v1_short"] = v1_short.fillna(False).to_numpy()
    ind["v2_long"] = v2_long.fillna(False).to_numpy()
    ind["v2_short"] = v2_short.fillna(False).to_numpy()
    return ind


# ---------------------------------------------------------------- A) 入场edge
def fwd_edge():
    sig = {"V0": ("buy_base", "sell_base"),
           "V1": ("v1_long", "v1_short"),
           "V2": ("v2_long", "v2_short")}
    acc = {k: {h: ([], []) for h in H} for k in sig}      # name->h->(long,short)
    base = {h: [] for h in H}
    for f in files:
        df = pd.read_parquet(f).sort_values("datetime").reset_index(drop=True)
        if len(df) < WARMUP + 200:
            continue
        ind = prep(df)
        c = ind["close"].to_numpy(); n = len(c); warm = np.arange(n) >= WARMUP
        for h in H:
            fwd = np.full(n, np.nan); fwd[:n - h] = c[h:] / c[:n - h] - 1
            m = ~np.isnan(fwd)
            base[h].append(fwd[warm & m])
            for k, (lc, sc) in sig.items():
                lb = ind[lc].to_numpy() & warm & m
                sb = ind[sc].to_numpy() & warm & m
                acc[k][h][0].append(fwd[lb]); acc[k][h][1].append(-fwd[sb])

    print("=" * 86)
    print("A) 入场信号前向收益(毛) vs 同向全bar基准  —— 多空合并超额>0 才说明入场有edge")
    print("=" * 86)
    print(f"{'变体':>4} {'h':>4} | {'多均值':>9}{'多胜率':>7} | {'空均值':>9}{'空胜率':>7} | "
          f"{'基准':>8} | {'多超额':>8} {'空超额':>8}")
    for k in sig:
        nl = sum(len(x) for x in acc[k][10][0]); ns = sum(len(x) for x in acc[k][10][1])
        for h in H:
            l = np.concatenate(acc[k][h][0]); s = np.concatenate(acc[k][h][1])
            a = np.concatenate(base[h])
            print(f"{k:>4} {h:>4} | {l.mean():>9.4%}{(l>0).mean():>7.1%} | "
                  f"{s.mean():>9.4%}{(s>0).mean():>7.1%} | {a.mean():>8.4%} | "
                  f"{l.mean()-a.mean():>8.4%} {s.mean()+a.mean():>8.4%}")
        print(f"     (多信号{nl}笔 / 空信号{ns}笔)")
        print("-" * 86)


# ---------------------------------------------------------------- B) 实现绩效
def run_reversion(ind):
    """V1: 极端偏离反向进场; 回到均值止盈, 继续极端止损, 超时离场。"""
    z = ind["z"].to_numpy(); c = ind["close"].to_numpy()
    n = len(c); warm = np.arange(n) >= WARMUP
    vl = ind["v1_long"].to_numpy(); vs = ind["v1_short"].to_numpy()
    trades = []; pos = 0; epx = 0.0; ei = -1
    for i in range(n):
        if not warm[i]:
            continue
        if pos > 0:
            ex = (z[i] >= 0) or (z[i] <= -Z_STOP) or (i - ei >= MAXH)
            if ex:
                trades.append((c[i] / epx - 1) - 2 * COST); pos = 0
        elif pos < 0:
            ex = (z[i] <= 0) or (z[i] >= Z_STOP) or (i - ei >= MAXH)
            if ex:
                trades.append((epx / c[i] - 1) - 2 * COST); pos = 0
        if pos == 0:
            if vl[i]:
                pos, epx, ei = 1, c[i], i
            elif vs[i]:
                pos, epx, ei = -1, c[i], i
    return trades


def run_momentum(ind):
    """V2: 去追高顺势进场; 仅在跌破MA250/反向时全平(让赢家跑, 无MAX_HOLD/无减仓)。"""
    c = ind["close"].to_numpy(); n = len(c); warm = np.arange(n) >= WARMUP
    vl = ind["v2_long"].to_numpy(); vs = ind["v2_short"].to_numpy()
    x_lf = ind["x_lf"].to_numpy(); x_sf = ind["x_sf"].to_numpy()
    trades = []; pos = 0; epx = 0.0
    for i in range(n):
        if not warm[i]:
            continue
        if pos > 0:
            if x_lf[i] or vs[i]:
                trades.append((c[i] / epx - 1) - 2 * COST); pos = 0
        elif pos < 0:
            if x_sf[i] or vl[i]:
                trades.append((epx / c[i] - 1) - 2 * COST); pos = 0
        if pos == 0:
            if vl[i]:
                pos, epx = 1, c[i]
            elif vs[i]:
                pos, epx = -1, c[i]
    return trades


def perf(rets):
    r = np.array(rets)
    if len(r) == 0:
        return {}
    w = r[r > 0]; l = r[r <= 0]
    pf = w.sum() / -l.sum() if l.sum() < 0 else np.inf
    return {"笔数": len(r), "胜率": (r > 0).mean(), "盈亏比":
            (w.mean() / -l.mean()) if len(l) else np.inf, "PF": pf,
            "等权合计": r.sum(), "单笔均": r.mean()}


def realized():
    rev = []; mom = []
    for f in files:
        df = pd.read_parquet(f).sort_values("datetime").reset_index(drop=True)
        if len(df) < WARMUP + 200:
            continue
        ind = prep(df)
        rev += run_reversion(ind); mom += run_momentum(ind)
    print("=" * 86)
    print("B) 实现绩效(各配合适出场, 单位仓, 成本2bp/边)")
    print("=" * 86)
    for name, r in [("V1 均值回归", rev), ("V2 动量去追高+让赢家跑", mom)]:
        p = perf(r)
        print(f"{name:<22} 笔数={p['笔数']:>5}  胜率={p['胜率']:.1%}  "
              f"盈亏比={p['盈亏比']:.2f}  PF={p['PF']:.2f}  "
              f"等权合计={p['等权合计']:.1%}  单笔均={p['单笔均']:.3%}")
    print("(对照) V0 原版带门: 胜率26.6% 盈亏比2.75 PF~1.0 等权合计~-14%")


if __name__ == "__main__":
    fwd_edge()
    realized()
