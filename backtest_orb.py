"""开盘区间突破(ORB)指标回测 —— 极简顺势 + 让利润奔跑, 按板块分组验证。

进场: 突破信号的下一根开盘; 止损: 开盘区间另一侧; 离场: 止损 / 会话末收盘
(让趋势日跑到收盘) / 可选 ATR 跟踪止损。成本按品种合约规格(tick+手续费+滑点)算。

用法:
    python backtest_orb.py
    python backtest_orb.py --slippage-ticks 0          # 被动限价
    python backtest_orb.py --trail-atr 2.0             # 启用跟踪止损
    python backtest_orb.py --products RB,CU,IF --or-bars 2 --rvol-min 1.5
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.indicators.opening_range_breakout import opening_range_breakout
from src.utils.contract_specs import PRODUCT_SPEC, cost_per_side, roundtrip_yuan

DATA_DIR = Path("data_futures/weighted_15min")

# 每板块代表品种
SECTOR_SETS = {
    "农产品":      ["C", "CS", "M", "Y", "OI"],
    "黑色/工业品": ["RB", "HC", "I", "J"],
    "有色":        ["CU", "AL", "ZN"],
    "贵金属":      ["AU", "AG"],
    "能化":        ["TA", "MA", "PP", "EG"],
    "股指":        ["IF", "IH", "IC", "IM"],
}


def simulate(df, sig, cost_side, trail_atr=0.0):
    """ORB 事件回测: 下一根开盘进场, 止损=区间另一侧, 会话末收盘了结(让利润奔跑)。"""
    o, h, l, c = (df['open'].to_numpy(), df['high'].to_numpy(),
                  df['low'].to_numpy(), df['close'].to_numpy())
    atr = sig['atr'].to_numpy()
    session = sig['session'].to_numpy()
    sig_arr = sig['signal'].to_numpy()
    stop_l = sig['stop_long'].to_numpy()
    stop_s = sig['stop_short'].to_numpy()
    n = len(df)

    trades = []
    i = 0
    while i < n - 1:
        s = sig_arr[i]
        if s == 0 or session[i + 1] != session[i]:  # 需同会话至少还有一根可进场
            i += 1
            continue

        entry = o[i + 1]
        cost = cost_side
        sess = session[i]

        if s == 1:
            stop = stop_l[i]
            if not np.isfinite(stop) or stop >= entry:
                i += 1
                continue
            risk = entry - stop
            eff_stop, exit_px, j = stop, None, i + 1
            for j in range(i + 1, n):
                if session[j] != sess:        # 安全网: 正常应在上一根会话末已了结
                    exit_px = c[j - 1]
                    break
                if l[j] <= eff_stop:
                    exit_px = eff_stop
                    break
                if j == n - 1 or session[j + 1] != sess:
                    exit_px = c[j]            # 会话末收盘了结
                    break
                if trail_atr > 0 and np.isfinite(atr[j]):
                    eff_stop = max(eff_stop, h[j] - trail_atr * atr[j])
            if exit_px is None:
                exit_px = c[j]
            pnl = exit_px - entry - 2 * cost
        else:
            stop = stop_s[i]
            if not np.isfinite(stop) or stop <= entry:
                i += 1
                continue
            risk = stop - entry
            eff_stop, exit_px, j = stop, None, i + 1
            for j in range(i + 1, n):
                if session[j] != sess:
                    exit_px = c[j - 1]
                    break
                if h[j] >= eff_stop:
                    exit_px = eff_stop
                    break
                if j == n - 1 or session[j + 1] != sess:
                    exit_px = c[j]
                    break
                if trail_atr > 0 and np.isfinite(atr[j]):
                    eff_stop = min(eff_stop, l[j] + trail_atr * atr[j])
            if exit_px is None:
                exit_px = c[j]
            pnl = entry - exit_px - 2 * cost

        trades.append((pnl, risk))
        i = j + 1

    if not trades:
        return {"trades": 0}
    pnl = np.array([t[0] for t in trades])
    risk = np.array([t[1] for t in trades])
    r = pnl / risk
    wins, losses = r[r > 0], r[r <= 0]
    gross_win = wins.sum() if len(wins) else 0.0
    gross_loss = -losses.sum() if len(losses) else 0.0
    equity = np.cumsum(r)
    peak = np.maximum.accumulate(equity)
    max_dd = (peak - equity).max() if len(equity) else 0.0
    return {
        "trades": len(r),
        "win_rate": len(wins) / len(r),
        "avg_win_R": wins.mean() if len(wins) else 0.0,
        "avg_loss_R": losses.mean() if len(losses) else 0.0,
        "payoff": (wins.mean() / -losses.mean()) if len(wins) and len(losses) else np.nan,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else np.inf,
        "expectancy_R": r.mean(),
        "total_R": r.sum(),
        "max_dd_R": max_dd,
    }


def run_product(p, slippage_ticks, or_bars, rvol_min, trail_atr):
    fp = DATA_DIR / f"{p}.parquet"
    if not fp.exists() or p not in PRODUCT_SPEC:
        return None
    df = pd.read_parquet(fp).reset_index(drop=True)
    sig = opening_range_breakout(df, or_bars=or_bars, rvol_min=rvol_min)
    res = simulate(df, sig, cost_side=cost_per_side(p, slippage_ticks), trail_atr=trail_atr)
    res["rt_yuan"] = roundtrip_yuan(p, slippage_ticks)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", type=str, default=None, help="指定品种(逗号分隔); 缺省=按板块全跑")
    ap.add_argument("--slippage-ticks", type=float, default=1.0, help="单边滑点(tick): 市价≈1, 被动限价≈0")
    ap.add_argument("--or-bars", type=int, default=2, help="开盘区间根数(2≈30min)")
    ap.add_argument("--rvol-min", type=float, default=1.2, help="突破放量下限")
    ap.add_argument("--trail-atr", type=float, default=0.0, help="ATR跟踪止损(0=关, 纯让利润跑到会话末)")
    args = ap.parse_args()

    print(f"ORB回测  滑点={args.slippage_ticks}tick  开盘区间={args.or_bars}根  "
          f"放量≥{args.rvol_min}  跟踪止损={args.trail_atr}ATR  离场=会话末/区间止损")
    print("=" * 108)
    print(f"{'品种':<6}{'板块':<12}{'往返成本':>9}{'交易':>6}{'胜率':>8}{'盈亏比':>8}"
          f"{'盈利因子':>9}{'期望(R)':>9}{'累计(R)':>9}{'最大回撤':>10}")
    print("-" * 108)

    if args.products:
        groups = {"自选": [p.strip().upper() for p in args.products.split(",") if p.strip()]}
    else:
        groups = SECTOR_SETS

    grand = []
    for sector, prods in groups.items():
        sec_rows = []
        for p in prods:
            res = run_product(p, args.slippage_ticks, args.or_bars, args.rvol_min, args.trail_atr)
            if res is None or res.get("trades", 0) == 0:
                continue
            sec_rows.append(res)
            grand.append(res)
            name = PRODUCT_SPEC[p]["name"]
            print(f"{p+' '+name:<6}{sector:<12}{res['rt_yuan']:>7.0f}元{res['trades']:>6}"
                  f"{res['win_rate']*100:>7.1f}%{res['payoff']:>8.2f}{res['profit_factor']:>9.2f}"
                  f"{res['expectancy_R']:>9.3f}{res['total_R']:>9.1f}{res['max_dd_R']:>10.1f}")
        if len(sec_rows) > 1:
            _subtotal(sector + " 小计", sec_rows)
        print("-" * 108)

    if grand:
        _subtotal("全部 汇总", grand)
        print("\n说明: ORB 是低胜率高盈亏比的顺势结构, 看的是盈利因子>1 与正期望, 而非胜率。")
        print("收益以 R(进场到区间止损)计。不同板块差异大, 重点看各板块小计与结构最顺的品种。")


def _subtotal(label, rows):
    tr = sum(r['trades'] for r in rows)
    w = sum(r['win_rate'] * r['trades'] for r in rows) / tr
    e = sum(r['expectancy_R'] * r['trades'] for r in rows) / tr
    pf_gw = sum(max(r['avg_win_R'], 0) * r['win_rate'] * r['trades'] for r in rows)
    tot = sum(r['total_R'] for r in rows)
    print(f"{label:<18}{'':>9}{tr:>6}{w*100:>7.1f}%{'':>8}{'':>9}{e:>9.3f}{tot:>9.1f}")


if __name__ == "__main__":
    main()
