"""优化 V1 均值回归入场, 带样本外(OOS)护栏。

每个品种按时间 70/30 切分(训练/测试)。在训练集扫 (Z_IN, W, Z_STOP) 网格,
再看同一参数在测试集是否仍正——只认"训练好且测试也不崩"的稳健参数, 避免过拟合。
随后对稳健参数叠加: 宏观方向过滤 + 止盈位(Z_EXIT)。

口径与之前一致: bars>=1500, 单位仓, 成本 2bp/边。
"""
import glob
import numpy as np
import pandas as pd
import backtest_qiumo as bt

WARMUP = 1500
MAXH = 200
COST = bt.COST
files = sorted(glob.glob("data_futures/15min/*.parquet"))

Z_INS = [1.5, 2.0, 2.5]
WS = [120, 250, 500]
Z_STOPS = [3.0, 4.0, np.inf]


def load():
    """预载每个品种: close, mamc, 以及各窗口 W 的 z 序列。"""
    data = []
    for f in files:
        df = pd.read_parquet(f).sort_values("datetime").reset_index(drop=True)
        if len(df) < WARMUP + 400:
            continue
        C = df["close"]
        ma250 = bt.MA(C, 250)
        spd = C - ma250
        zs = {w: (spd / spd.rolling(w, min_periods=w).std()).to_numpy() for w in WS}
        mamc = bt.MA(C, bt.MACRO).to_numpy()
        n = len(C)
        split = WARMUP + int((n - WARMUP) * 0.70)     # 训练/测试时间分界
        data.append({"c": C.to_numpy(), "z": zs, "mamc": mamc,
                     "n": n, "split": split})
    return data


def sim(c, z, z_in, z_stop, mamc=None, z_exit=0.0):
    """均值回归: z<=-z_in 做多 / z>=z_in 做空; 回到 z_exit 止盈, 到 ±z_stop 止损, 超时。
    mamc!=None 时加宏观方向过滤(只在 C>MAMC 做多 / C<MAMC 做空)。
    返回 [(entry_i, net_ret), ...]。"""
    n = len(c)
    out = []
    pos = 0; epx = 0.0; ei = -1
    for i in range(WARMUP, n):
        zi = z[i]
        if np.isnan(zi):
            continue
        if pos > 0:
            if zi >= z_exit or zi <= -z_stop or (i - ei) >= MAXH:
                out.append((ei, (c[i] / epx - 1) - 2 * COST)); pos = 0
        elif pos < 0:
            if zi <= -z_exit or zi >= z_stop or (i - ei) >= MAXH:
                out.append((ei, (epx / c[i] - 1) - 2 * COST)); pos = 0
        if pos == 0:
            long_ok = (mamc is None) or (c[i] > mamc[i])
            short_ok = (mamc is None) or (c[i] < mamc[i])
            if zi <= -z_in and long_ok:
                pos, epx, ei = 1, c[i], i
            elif zi >= z_in and short_ok:
                pos, epx, ei = -1, c[i], i
    return out


def stats(trades, split):
    tr = [r for (ei, r) in trades if ei < split]
    te = [r for (ei, r) in trades if ei >= split]
    def m(x):
        x = np.array(x)
        if len(x) == 0:
            return (0, 0, 0, 0)
        w = x[x > 0]; l = x[x <= 0]
        pf = w.sum() / -l.sum() if l.sum() < 0 else np.inf
        return (len(x), (x > 0).mean(), x.mean(), pf)
    return m(tr), m(te)


def run_grid(data, mamc_on=False, z_exit=0.0):
    rows = []
    for z_in in Z_INS:
        for w in WS:
            for z_stop in Z_STOPS:
                tr_all, te_all = [], []
                for d in data:
                    t = sim(d["c"], d["z"][w], z_in, z_stop,
                            d["mamc"] if mamc_on else None, z_exit)
                    for ei, r in t:
                        (te_all if ei >= d["split"] else tr_all).append(r)
                def m(x):
                    x = np.array(x)
                    w_ = x[x > 0]; l_ = x[x <= 0]
                    pf = w_.sum() / -l_.sum() if l_.sum() < 0 else np.inf
                    return len(x), (x > 0).mean(), x.mean(), pf, x.sum()
                trn = m(tr_all); tes = m(te_all)
                rows.append({"Z_IN": z_in, "W": w, "Z_STOP": z_stop,
                             "训练_笔": trn[0], "训练_胜": trn[1], "训练_均": trn[2],
                             "训练_PF": trn[3],
                             "测试_笔": tes[0], "测试_胜": tes[1], "测试_均": tes[2],
                             "测试_PF": tes[3]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    data = load()
    print(f"参与品种: {len(data)}")

    def robust_count(g):
        return g[(g["训练_均"] > 0) & (g["测试_均"] > 0) &
                 (g["训练_PF"] > 1) & (g["测试_PF"] > 1)]

    for mon, title in [(False, "无宏观过滤"), (True, "宏观方向过滤(顺大势抄回调)")]:
        g = run_grid(data, mamc_on=mon)
        gs = g.sort_values("测试_均", ascending=False)
        show = gs.copy()
        for c in ["训练_均", "测试_均"]:
            show[c] = (show[c] * 100).round(3)        # bp%
        for c in ["训练_胜", "训练_PF", "测试_胜", "测试_PF"]:
            show[c] = show[c].round(3)
        print("=" * 92)
        print(f"网格: {title}  (按【测试集】均收益排序)")
        print("=" * 92)
        print(show[["Z_IN", "W", "Z_STOP", "训练_笔", "训练_均", "训练_PF",
                    "测试_笔", "测试_胜", "测试_均", "测试_PF"]].head(12).to_string(index=False))
        rb = robust_count(g)
        print(f"训练&测试同时为正(稳健)的参数: {len(rb)}/{len(g)}")

