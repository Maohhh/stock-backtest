"""期货求魔·趋势过滤版 V2 —— 回测引擎。

把同花顺(THS)主图指标逐行翻译为 Python，并在 15 分钟加权数据上做事件驱动回测。

离场建模(按指标原意「三档分批」):
    多头持仓 = 3/3 单位:
      · 极平(EXT_L, 极端价差且 SPD>0) -> 减 1/3
      · 减仓(X_LW, 收盘跌破 MA120)     -> 减 1/3
      · 平多(X_LF, 收盘跌破 MA250)     -> 清剩余
      · 触止损(LOW<=STOP_L)            -> 全平(止损价成交)
      · 反向 SELL_BASE / 持仓超 MAX_HOLD -> 全平
    空头对称。

执行: 信号按收盘价计算，于该 K 线收盘成交；每边收取 COST 成本(默认 2bp)。
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

# ===== 参数(默认对 15 分钟，与指标一致) =====
MACRO = 1500
SLP_N = 10
SLP_E = 0.0001
K = 3
NEARP = 0.4
SL_BUF = 0.001
ARMED_WIN = 65
MAX_HOLD = 200
EXT_N = 12000
COST = 0.0002          # 单边成本(滑点+手续), 2bp


# --------------------------------------------------------------------------
# THS 函数语义实现
# --------------------------------------------------------------------------
def MA(s, n):       return s.rolling(n, min_periods=n).mean()
def REF(s, n):      return s.shift(n)
def LLV(s, n):      return s.rolling(n, min_periods=n).min()
def HHV(s, n):      return s.rolling(n, min_periods=n).max()


def CROSS(a, b):
    """a 上穿 b: 前一根 a<b 且当根 a>b。"""
    a, b = pd.Series(a), pd.Series(b)
    return (a.shift(1) < b.shift(1)) & (a > b)


def BARSLAST(cond):
    """距上次 cond 为真的 K 线数(当根为真则 0); 从未为真填 +inf。"""
    cond = np.asarray(cond, dtype=bool)
    idx = np.where(cond, np.arange(len(cond)), np.nan)
    idx = pd.Series(idx).ffill().to_numpy()
    out = np.arange(len(cond), dtype=float) - idx
    out[np.isnan(idx)] = np.inf
    return pd.Series(out)


def VALUEWHEN(cond, x):
    """cond 最近一次为真时 x 的取值。"""
    cond = pd.Series(cond).to_numpy(dtype=bool)
    x = pd.Series(x).reset_index(drop=True)
    return x.where(cond).ffill()


# --------------------------------------------------------------------------
# 指标计算(逐行对应 THS 源码)
# --------------------------------------------------------------------------
def compute_indicators(df: pd.DataFrame, use_macro: bool = True) -> pd.DataFrame:
    C, H, L = df["close"], df["high"], df["low"]

    ma20 = MA(C, 20)
    ma120 = MA(C, 120)
    ma250 = MA(C, 250)
    mamc = MA(C, MACRO)

    if use_macro:
        bull = C > mamc
        bear = C < mamc
    else:
        # 去掉宏观趋势门: 门恒开(回到原版「期货求魔」)
        bull = pd.Series(True, index=C.index)
        bear = pd.Series(True, index=C.index)

    slp = (ma250 - REF(ma250, SLP_N)) / C
    upt = slp > SLP_E
    dnt = slp < -SLP_E
    rng = (~upt) & (~dnt)

    pl = REF(L, K) == LLV(L, 2 * K + 1)
    ph = REF(H, K) == HHV(H, 2 * K + 1)
    ma250k = REF(ma250, K)
    nl = pl & (((REF(L, K) - ma250k).abs() / ma250k * 100) <= NEARP)
    nh = ph & (((REF(H, K) - ma250k).abs() / ma250k * 100) <= NEARP)

    nl_held = nl & (REF(L, K) >= ma250k)
    nl_broke = nl & (REF(L, K) < ma250k)
    nh_held = nh & (REF(H, K) <= ma250k)
    nh_broke = nh & (REF(H, K) > ma250k)

    pre_high_cur = REF(HHV(H, 20), K + 1)
    pre_low_cur = REF(LLV(L, 20), K + 1)

    nl_evt = (nl_held | nl_broke) & (pre_high_cur > ma250k)
    nh_evt = (nh_held | nh_broke) & (pre_low_cur < ma250k)

    armed_pre_h = VALUEWHEN(nl_evt, pre_high_cur)
    armed_pre_l = VALUEWHEN(nh_evt, pre_low_cur)

    armed_sl_l = VALUEWHEN(nl_evt, REF(L, K))
    armed_sl_h = VALUEWHEN(nh_evt, REF(H, K))
    stop_l = armed_sl_l * (1 - SL_BUF)
    stop_s = armed_sl_h * (1 + SL_BUF)

    armed_long = BARSLAST(nl_evt) <= ARMED_WIN
    armed_short = BARSLAST(nh_evt) <= ARMED_WIN
    last_nl_held = VALUEWHEN(nl_evt, nl_held.astype(float)) >= 1
    last_nh_held = VALUEWHEN(nh_evt, nh_held.astype(float)) >= 1

    abuy = armed_long & (C > armed_pre_h) & (C > ma250) & (upt | rng) & bull
    asell = armed_short & (C < armed_pre_l) & (C < ma250) & (dnt | rng) & bear

    s2_long_ok = REF(C, 1) > REF(ma250, 1)
    s2_short_ok = REF(C, 1) < REF(ma250, 1)
    s2_cond_l = abuy & (~last_nl_held) & s2_long_ok
    s2_cond_s = asell & (~last_nh_held) & s2_short_ok

    buy_s1 = abuy & last_nl_held & (~REF(abuy, 1).fillna(False).astype(bool))
    buy_s2 = s2_cond_l & (~REF(s2_cond_l, 1).fillna(False).astype(bool))
    buy_base = buy_s1 | buy_s2

    sell_s1 = asell & last_nh_held & (~REF(asell, 1).fillna(False).astype(bool))
    sell_s2 = s2_cond_s & (~REF(s2_cond_s, 1).fillna(False).astype(bool))
    sell_base = sell_s1 | sell_s2

    x_lw = CROSS(ma120, C)      # 跌破 MA120 -> 减仓
    x_lf = CROSS(ma250, C)      # 跌破 MA250 -> 平多
    x_sw = CROSS(C, ma120)      # 升破 MA120 -> 减仓(空)
    x_sf = CROSS(C, ma250)      # 升破 MA250 -> 平空

    spd = C - ma250
    # THS 在 K 线不足 EXT_N 时按现有全部 K 线取极值，故 min_periods=1
    ext_abs = spd.abs()
    ext_hi = ext_abs == ext_abs.rolling(EXT_N, min_periods=1).max()

    out = df.copy()
    out["ma120"], out["ma250"], out["mamc"] = ma120, ma250, mamc
    out["buy_base"] = buy_base.fillna(False).to_numpy()
    out["sell_base"] = sell_base.fillna(False).to_numpy()
    out["x_lw"] = x_lw.fillna(False).to_numpy()
    out["x_lf"] = x_lf.fillna(False).to_numpy()
    out["x_sw"] = x_sw.fillna(False).to_numpy()
    out["x_sf"] = x_sf.fillna(False).to_numpy()
    out["stop_l"] = stop_l.to_numpy()
    out["stop_s"] = stop_s.to_numpy()
    out["ext_hi"] = ext_hi.fillna(False).to_numpy()
    out["spd"] = spd.to_numpy()
    out["warm"] = mamc.notna().to_numpy()      # 预热完成(MA1500 有效)
    return out


# --------------------------------------------------------------------------
# 事件驱动持仓模拟(三档分批)
# --------------------------------------------------------------------------
def simulate(df: pd.DataFrame, product: str) -> list[dict]:
    trades = []
    pos = 0          # +1 多 / -1 空 / 0 空仓
    thirds = 0       # 剩余仓位(以 1/3 计, 0..3)
    entry_px = 0.0
    entry_i = -1
    realized_thirds = 0

    rows = df.reset_index(drop=True)
    n = len(rows)

    def close_part(i, px, k, reason):
        """平掉 k 个三分之一仓位，记录一笔。"""
        nonlocal thirds, realized_thirds
        k = min(k, thirds)
        if k <= 0:
            return
        frac = k / 3.0
        ret = (px / entry_px - 1.0) * pos      # 方向收益率
        ret -= COST * 2                         # 进+出双边成本(按该份额计一次往返)
        trades.append({
            "product": product, "dir": "多" if pos > 0 else "空",
            "entry_i": entry_i, "exit_i": i,
            "entry_dt": rows["datetime"].iloc[entry_i],
            "exit_dt": rows["datetime"].iloc[i],
            "entry_px": entry_px, "exit_px": px,
            "frac": frac, "hold_bars": i - entry_i,
            "ret": ret * frac,                  # 对整仓的贡献
            "ret_unit": ret,                    # 该份额自身收益率
            "reason": reason,
        })
        thirds -= k

    def flat(i, px, reason):
        nonlocal pos, thirds
        if thirds > 0:
            close_part(i, px, thirds, reason)
        pos, thirds = 0, 0

    for i in range(n):
        r = rows.iloc[i]
        if not r["warm"]:
            continue

        # ---- 持仓中: 先判止损(盘中), 再判离场事件(收盘) ----
        if pos > 0:
            if not np.isnan(r["stop_l"]) and r["low"] <= r["stop_l"]:
                flat(i, r["stop_l"], "止损")
            elif r["x_lf"]:
                flat(i, r["close"], "平多")
            elif r["sell_base"]:
                flat(i, r["close"], "反向")
            elif (i - entry_i) >= MAX_HOLD:
                flat(i, r["close"], "超时")
            else:
                if r["ext_hi"] and r["spd"] > 0:
                    close_part(i, r["close"], 1, "极平")
                if r["x_lw"] and thirds > 0:
                    close_part(i, r["close"], 1, "减仓")
                if thirds == 0:
                    pos = 0
        elif pos < 0:
            if not np.isnan(r["stop_s"]) and r["high"] >= r["stop_s"]:
                flat(i, r["stop_s"], "止损")
            elif r["x_sf"]:
                flat(i, r["close"], "平空")
            elif r["buy_base"]:
                flat(i, r["close"], "反向")
            elif (i - entry_i) >= MAX_HOLD:
                flat(i, r["close"], "超时")
            else:
                if r["ext_hi"] and r["spd"] < 0:
                    close_part(i, r["close"], 1, "极平")
                if r["x_sw"] and thirds > 0:
                    close_part(i, r["close"], 1, "减仓")
                if thirds == 0:
                    pos = 0

        # ---- 空仓: 开新仓(含反向后同根开仓) ----
        if pos == 0:
            if r["buy_base"]:
                pos, thirds, entry_px, entry_i = 1, 3, r["close"], i
            elif r["sell_base"]:
                pos, thirds, entry_px, entry_i = -1, 3, r["close"], i

    return trades


# --------------------------------------------------------------------------
# 绩效统计
# --------------------------------------------------------------------------
def summarize(trades: list[dict]) -> dict:
    if not trades:
        return {}
    t = pd.DataFrame(trades)
    # 以"整仓收益率"口径: 一笔完整交易的收益 = 各份额 ret_unit*frac 之和
    by_entry = t.groupby(["product", "entry_i"]).agg(
        ret=("ret", "sum"), dir=("dir", "first"),
        hold=("hold_bars", "max")).reset_index()
    rets = by_entry["ret"]
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    gross_win = wins.sum()
    gross_loss = -losses.sum()
    return {
        "trades": len(by_entry),
        "win_rate": len(wins) / len(by_entry),
        "avg_ret": rets.mean(),
        "total_ret": rets.sum(),
        "pf": (gross_win / gross_loss) if gross_loss > 0 else np.inf,
        "avg_hold": by_entry["hold"].mean(),
        "best": rets.max(), "worst": rets.min(),
    }


def main():
    ap = argparse.ArgumentParser(description="期货求魔V2 回测")
    ap.add_argument("--datadir", default="data_futures/15min")
    ap.add_argument("--products", default=None, help="逗号分隔, 默认全部")
    ap.add_argument("--out", default="data_futures/backtest_qiumo")
    ap.add_argument("--no-macro", action="store_true",
                    help="去掉宏观趋势门(回到原版期货求魔)")
    args = ap.parse_args()
    use_macro = not args.no_macro
    out_dir = args.out if use_macro else args.out + "_nomacro"

    files = sorted(glob.glob(os.path.join(args.datadir, "*.parquet")))
    if args.products:
        keep = {p.strip().upper() for p in args.products.split(",")}
        files = [f for f in files if os.path.basename(f)[:-8] in keep]

    all_trades = []
    rows = []
    for f in files:
        product = os.path.basename(f)[:-8]
        df = pd.read_parquet(f).sort_values("datetime").reset_index(drop=True)
        ind = compute_indicators(df, use_macro=use_macro)
        warm_bars = int(ind["warm"].sum())
        if warm_bars < 50:
            rows.append({"product": product, "bars": len(df),
                         "warm_bars": warm_bars, "trades": 0,
                         "note": "预热不足"})
            continue
        trades = simulate(ind, product)
        all_trades.extend(trades)
        s = summarize(trades)
        rows.append({"product": product, "bars": len(df),
                     "warm_bars": warm_bars, **s})

    os.makedirs(out_dir, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, "by_product.csv"), index=False)
    if all_trades:
        pd.DataFrame(all_trades).to_csv(
            os.path.join(out_dir, "trades.csv"), index=False)

    # 汇总
    traded = res[res.get("trades", 0).fillna(0) > 0] if "trades" in res else res
    print("=" * 72)
    print("期货求魔 回测结果  [宏观门: %s]" % ("开" if use_macro else "关/原版"))
    print("=" * 72)
    if all_trades:
        agg = summarize(all_trades)
        print(f"参与品种(有交易): {len(traded)}   总交易数: {agg['trades']}")
        print(f"胜率: {agg['win_rate']:.1%}   盈亏比(PF): {agg['pf']:.2f}")
        print(f"单笔均收益: {agg['avg_ret']:.2%}   合计(等权单位): {agg['total_ret']:.1%}")
        print(f"平均持仓: {agg['avg_hold']:.0f} 根   最好/最差单笔: "
              f"{agg['best']:.1%} / {agg['worst']:.1%}")
        print("-" * 72)
        show = traded.sort_values("total_ret", ascending=False)
        cols = ["product", "bars", "warm_bars", "trades", "win_rate",
                "total_ret", "pf", "avg_hold"]
        with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
            print(show[cols].to_string(index=False))
    else:
        print("无交易(预热不足或无信号)")
    print("=" * 72)
    print(f"明细已写入: {out_dir}/")


if __name__ == "__main__":
    main()
