"""日内趋势同向均值回归指标的回测验证。

用 data_futures/weighted_15min 的持仓量加权连续数据, 在几个代表不同底层逻辑的
品种上(工业品/能化/有色/贵金属/农产品/股指)分别回测, 输出胜率/盈亏比/盈利因子/
最大回撤/净期望(已扣成本)。

注意: 不同板块底层逻辑不同, 不要求每个品种都好。本脚本的目的是验证 edge 在
"该有的地方"是否真实存在, 以及成本是否把它吃光。

用法:
    python backtest_intraday_reversion.py
    python backtest_intraday_reversion.py --products RB,M,CU,AU,TA,IF --cost-bps 4
    python backtest_intraday_reversion.py --max-hold 16 --trend-align
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.indicators.intraday_reversion import intraday_reversion
from src.utils.contract_specs import PRODUCT_SPEC, cost_per_side, roundtrip_yuan

DATA_DIR = Path("data_futures/weighted_15min")

DEFAULT_PRODUCTS = ["C", "CS", "RB", "M", "CU", "AU", "TA", "IF"]


def simulate(df: pd.DataFrame, sig: pd.DataFrame, cost_side: float, max_hold: int,
             tp_atr: float = 0.6, sl_atr: float = 1.6) -> dict:
    """逐信号事件回测: 下一根开盘进场, 触止盈/止损/超时离场, 扣双边成本。

    高胜率反转的正确离场结构 = *近止盈 + 宽止损*:
      止盈 = 进场 ± tp_atr*ATR (吃回抽的一段, 近 -> 命中率高);
      止损 = 进场 ∓ sl_atr*ATR (给噪声留空间 -> 不被针尖回抽轻易扫掉)。
    天然负偏度(多次小赢、偶尔大亏), 用 regime+趋势同向过滤压制尾部。
    收益以 R(初始风险=sl_atr*ATR)计量, 跨品种可比、与点值无关。
    cost_side: 单边成本(价格单位, 与 pnl 同单位), = 手续费/手/乘数 + 滑点tick数*tick。
    """
    o = df['open'].to_numpy()
    h = df['high'].to_numpy()
    l = df['low'].to_numpy()
    c = df['close'].to_numpy()
    atr = sig['atr'].to_numpy()
    sig_arr = sig['signal'].to_numpy()
    n = len(df)

    trades = []
    i = 0
    while i < n - 1:
        s = sig_arr[i]
        if s == 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
            i += 1
            continue

        entry = o[i + 1]  # 下一根开盘进场
        cost = cost_side  # 单边成本(价格单位)
        a = atr[i]
        risk = sl_atr * a
        last = min(i + max_hold, n - 1)

        if s == 1:  # 做多: 买恐慌, 近止盈 + 宽止损
            tp, stop = entry + tp_atr * a, entry - sl_atr * a
            outcome, j = None, i + 1
            for j in range(i + 1, last + 1):
                if l[j] <= stop:            # 同根保守: 先判止损
                    outcome = stop - entry
                    break
                if h[j] >= tp:
                    outcome = tp - entry
                    break
            if outcome is None:
                outcome = c[last] - entry   # 超时按收盘
        else:  # 做空: 卖贪婪
            tp, stop = entry - tp_atr * a, entry + sl_atr * a
            outcome, j = None, i + 1
            for j in range(i + 1, last + 1):
                if h[j] >= stop:
                    outcome = entry - stop
                    break
                if l[j] <= tp:
                    outcome = entry - tp
                    break
            if outcome is None:
                outcome = entry - c[last]

        pnl = outcome - 2 * cost
        trades.append((pnl, risk))
        i = j + 1

    if not trades:
        return {"trades": 0}

    pnl = np.array([t[0] for t in trades])
    risk = np.array([t[1] for t in trades])
    r = pnl / risk  # 以 R 计

    wins = r[r > 0]
    losses = r[r <= 0]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", type=str, default=",".join(DEFAULT_PRODUCTS))
    ap.add_argument("--slippage-ticks", type=float, default=1.0,
                    help="单边滑点(以tick计)。市价单约1, 限价被动成交约0, 这是短线成本主因")
    ap.add_argument("--max-hold", type=int, default=16, help="最长持仓根数(15min*16≈4小时)")
    ap.add_argument("--no-trend-align", action="store_true", help="关闭大周期趋势同向(双向 fade)")
    ap.add_argument("--stretch-thr", type=float, default=1.2)
    ap.add_argument("--tp-atr", type=float, default=0.6, help="止盈=tp_atr*ATR(近, 高命中)")
    ap.add_argument("--sl-atr", type=float, default=1.6, help="止损=sl_atr*ATR(宽, 扛噪声)")
    args = ap.parse_args()

    products = [p.strip().upper() for p in args.products.split(",") if p.strip()]
    trend_align = not args.no_trend_align

    print(f"回测: 趋势同向={trend_align}  单边滑点={args.slippage_ticks}tick  最长持仓={args.max_hold}根  "
          f"止盈={args.tp_atr}ATR  止损={args.sl_atr}ATR")
    print("成本 = 往返手续费/手/乘数 + 单边滑点*tick (按品种合约规格, 非价格基点)")
    print("=" * 116)
    header = (f"{'品种':<6}{'板块':<12}{'往返成本':>10}{'交易数':>6}{'胜率':>8}{'盈亏比':>8}"
              f"{'盈利因子':>9}{'期望(R)':>9}{'累计(R)':>9}{'最大回撤(R)':>11}")
    print(header)
    print("-" * 116)

    rows = []
    for p in products:
        fp = DATA_DIR / f"{p}.parquet"
        spec = PRODUCT_SPEC.get(p)
        if not fp.exists() or spec is None:
            print(f"{p:<6}(无数据或无合约规格, 跳过)")
            continue
        df = pd.read_parquet(fp).reset_index(drop=True)
        # 单边成本(价格单位) = 手续费每边/乘数 + 滑点tick数*tick (共享自 contract_specs)
        cost_side = cost_per_side(p, args.slippage_ticks)
        rt_yuan = roundtrip_yuan(p, args.slippage_ticks)  # 往返成本(元/手), 用于展示
        sig = intraday_reversion(df, stretch_thr=args.stretch_thr, trend_align=trend_align)
        res = simulate(df, sig, cost_side=cost_side, max_hold=args.max_hold,
                       tp_atr=args.tp_atr, sl_atr=args.sl_atr)
        sector = spec['sector']
        if res.get("trades", 0) == 0:
            print(f"{p:<6}{sector:<12}{'':>10}{'0':>6}  (无触发信号)")
            continue
        rows.append((p, res))
        print(f"{p:<6}{sector:<12}{rt_yuan:>8.1f}元{res['trades']:>6}{res['win_rate']*100:>7.1f}%"
              f"{res['payoff']:>8.2f}{res['profit_factor']:>9.2f}"
              f"{res['expectancy_R']:>9.3f}{res['total_R']:>9.1f}{res['max_dd_R']:>11.1f}")

    if rows:
        print("-" * 116)
        all_tr = sum(r['trades'] for _, r in rows)
        w = sum(r['win_rate'] * r['trades'] for _, r in rows) / all_tr
        e = sum(r['expectancy_R'] * r['trades'] for _, r in rows) / all_tr
        tot = sum(r['total_R'] for _, r in rows)
        print(f"{'汇总':<6}{'(加权)':<12}{'':>10}{all_tr:>6}{w*100:>7.1f}%"
              f"{'':>8}{'':>9}{e:>9.3f}{tot:>9.1f}")
        print("\n说明: 收益以 R(单笔初始风险)计。短线成本主因是滑点(以tick计)而非手续费:")
        print("市价单进出≈1tick/边, 被动限价单≈0 但会面临成交不全与逆向选择。")


if __name__ == "__main__":
    main()
