#!/usr/bin/env python3
"""
15分钟「顺势买回调」高胜率试验：LON/MACD 定方向 + KDJ 抓超卖回调

思路（多头，空头镜像）：
    方向过滤：LON>0 且 MACD 双线在 0 轴上方（处于上涨结构）；
    入场：KDJ 在超卖区(J<20)且 K 上穿 D（回调结束的拐点）；
    出场：KDJ 超买(J>80)或 K 下穿 D（回调反弹到位）。
这是典型「顺势买跌」均值回归，胜率天然偏高；重点看扣手续费后的真实期望。

数据：工业品篮子 15min 持仓量加权连续。
用法： python lon_macd_kdj15.py
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.indicators import macd as macd_ind, lon as lon_ind, kdj as kdj_ind  # noqa: E402
from src.backtest.futures_engine import run_backtest  # noqa: E402

DIR15 = os.path.join("data_futures", "weighted_15min")
DATA_REF = "origin/claude/futures-data-inventory-fdwz31"
OUT_DIR = "lon_macd_results"
COMMISSION = 0.0003
WARMUP = 60
BASKET = ["AU", "AG", "AL", "CU", "NI", "PB", "SN", "ZN", "RB", "HC", "I", "JM",
          "SF", "SM", "SS", "TA", "MA", "PP", "L", "V", "EG", "FU", "BU", "SC", "RU", "FG"]


def load15(sym):
    path = os.path.join(DIR15, f"{sym}.parquet")
    if not os.path.exists(path):
        os.makedirs(DIR15, exist_ok=True)
        rel = f"data_futures/weighted_15min/{sym}.parquet"
        try:
            c = subprocess.check_output(["git", "show", f"{DATA_REF}:{rel}"], stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            return None
        open(path, "wb").write(c)
    df = pd.read_parquet(path).rename(columns={"datetime": "date"})
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df[df["close"] > 0].reset_index(drop=True)


def signals(df):
    m = macd_ind(df); dif = m["DIF"].to_numpy(); dea = m["DEA"].to_numpy()
    lon = lon_ind(df)["LON"].to_numpy()
    k = kdj_ind(df); K = k["K"].to_numpy(); D = k["D"].to_numpy(); J = k["J"].to_numpy()
    pK = np.r_[np.nan, K[:-1]]; pD = np.r_[np.nan, D[:-1]]
    kup = (K > D) & (pK <= pD); kdn = (K < D) & (pK >= pD)
    up = (lon > 0) & (dif > 0) & (dea > 0)
    dn = (lon < 0) & (dif < 0) & (dea < 0)
    le = np.nan_to_num(up & kup & (J < 20)).astype(bool)
    se = np.nan_to_num(dn & kdn & (J > 80)).astype(bool)
    lx = np.nan_to_num((J > 80) | kdn).astype(bool)
    sx = np.nan_to_num((J < 20) | kup).astype(bool)
    n = len(df); st = np.zeros(n, np.int8); s = 0
    for t in range(WARMUP, n):
        if s == 0:
            if le[t]:
                s = 1
            elif se[t]:
                s = -1
        elif s == 1:
            if lx[t]:
                s = 0
        else:
            if sx[t]:
                s = 0
        st[t] = s
    return st


def main():
    syms = [s for s in BASKET if load15(s) is not None]
    daily = {}; rows = []; ann = None
    for s in syms:
        df = load15(s)
        if df is None or len(df) < 500:
            continue
        if ann is None:
            ann = int(len(df) / ((df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25))
        st = signals(df)
        r = run_backtest(pd.DataFrame({"date": df["date"], "close": df["close"].values, "state": st}),
                         commission=COMMISSION, annualization=ann)
        daily[s] = pd.Series(r["equity_curve"]["equity"].pct_change().fillna(0).values,
                             index=pd.to_datetime(df["date"].values))
        rows.append({"品种": s, "总收益%": round(r["total_return"]*100, 1), "夏普": round(r["sharpe"], 2),
                     "交易": r["n_trades"], "胜率%": round(r["win_rate"]*100, 1),
                     "盈亏比": round(r["profit_factor"], 2),
                     "平均赢%": round(r["avg_win"]*100, 2), "平均亏%": round(r["avg_loss"]*100, 2)})
    t = pd.DataFrame(rows)
    mat = pd.DataFrame(daily).sort_index(); pr = mat.mean(axis=1, skipna=True).fillna(0)
    eq = (1+pr).cumprod()
    sh = pr.mean()/pr.std()*np.sqrt(ann) if pr.std() else 0
    cagr = eq.iloc[-1]**(1/(len(eq)/ann))-1 if eq.iloc[-1] > 0 else -1
    mdd = (eq/eq.cummax()-1).min()

    print("# 15min 顺势买回调(LON/MACD方向 + KDJ超卖)\n")
    print(t.sort_values("总收益%", ascending=False).to_string(index=False))
    print(f"\n=== 等权组合 ===  年化 {cagr*100:.1f}%  夏普 {sh:.2f}  回撤 {mdd*100:.1f}%")
    print(f"平均单品种胜率 {t['胜率%'].mean():.1f}%  平均盈亏比 {t['盈亏比'].mean():.2f}  "
          f"盈利品种 {(t['总收益%']>0).sum()}/{len(t)}")
    print(f"平均赢 {t['平均赢%'].mean():.2f}%  平均亏 {t['平均亏%'].mean():.2f}%")
    # 期望(每笔, 名义%)
    exp = t["胜率%"].mean()/100*t["平均赢%"].mean() + (1-t["胜率%"].mean()/100)*t["平均亏%"].mean()
    print(f"近似每笔期望 {exp:.3f}%（已含手续费；为正才真赚钱）")
    t.to_csv(os.path.join(OUT_DIR, "kdj15.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
