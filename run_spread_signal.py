"""
跨品种套利篮子 —— 今日信号面板（每天收盘后运行一次）

下载最新日线连续主力，计算每个套利对的当前 Z 分数，直接给出**今天该做什么**：
开仓 / 持有 / 止盈 / 止损，以及按你账户资金的建议手数。

用法: python run_spread_signal.py [账户资金, 默认200000]
"""

import sys

import numpy as np
import pandas as pd

from src.data.futures import download_daily
from src.indicators.spread_zscore import build_spread, spread_zscore, reversion_position
from src.backtest.spread_money import SpreadSpec, SPEC, _unit_margin_yuan

# 套利对：(名称, 主力代码A, 主力代码B, 单腿品种A, 单腿品种B, 价差模式, 手数A, 手数B)
# 全部为券商挂牌单一套利标的, 交易所对冲比例 1:1, diff 价差
PAIRS = [
    ("玉米-淀粉", "C0", "CS0", "C", "CS", "diff", 1, 1),
    ("PVC-聚丙烯", "V0", "PP0", "V", "PP", "diff", 1, 1),
    ("棉花-棉纱", "CF0", "CY0", "CF", "CY", "diff", 1, 1),
]
WINDOW, ENTRY, EXIT, STOP = 30, 2.0, 0.5, 4.0


def latest_close(symbol):
    df = download_daily(symbol)
    if df is None:
        return None
    return pd.to_numeric(df["close"], errors="coerce")


def action_for(z_now, pos_now):
    """根据当前 z 与持仓状态，给出今日动作文字。"""
    if pos_now == 0:
        if z_now > ENTRY:
            return "★ 开仓：做空价差（卖B买A）", f"z={z_now:.2f} > {ENTRY}"
        if z_now < -ENTRY:
            return "★ 开仓：做多价差（买B卖A）", f"z={z_now:.2f} < -{ENTRY}"
        return "观望（无信号）", f"|z|={abs(z_now):.2f} < {ENTRY}"
    if pos_now == 1:   # 持多价差
        return f"持有多价差｜止盈 z≥-{EXIT}｜止损 z<-{STOP}", f"当前 z={z_now:.2f}"
    return f"持有空价差｜止盈 z≤+{EXIT}｜止损 z>+{STOP}", f"当前 z={z_now:.2f}"


def main():
    capital = float(sys.argv[1]) if len(sys.argv) > 1 else 200000.0
    alloc = capital / len(PAIRS)
    print("=" * 74)
    print(f"跨品种套利 · 今日信号  ({pd.Timestamp.now().date()})  账户 ¥{capital:,.0f}")
    print("=" * 74)
    for name, ca, cb, pa, pb, mode, la, lb in PAIRS:
        a, b = latest_close(ca), latest_close(cb)
        if a is None or b is None:
            print(f"{name}: 数据获取失败"); continue
        df = pd.DataFrame({"a": a, "b": b}).dropna()
        sp = build_spread(df["a"], df["b"], mode if mode == "ratio" else "diff")
        if mode == "ratio":
            sp = (np.log(df["b"]) - np.log(df["a"])) * 10000
        z = spread_zscore(sp, WINDOW)
        pos = reversion_position(z, ENTRY, EXIT, STOP)
        z_now, pos_now = z.iloc[-1], int(pos.iloc[-1])
        act, why = action_for(z_now, pos_now)
        # 建议手数
        margin = _unit_margin_yuan(SpreadSpec(name, pa, pb, la, lb, mode),
                                   df["a"].iloc[-1], df["b"].iloc[-1])
        units = max(1, int(alloc * 0.5 / margin)) if margin > 0 else 1
        print(f"\n【{name}】 {ca} / {cb}   截至 {df.index[-1].date()}")
        print(f"   价差={sp.iloc[-1]:.1f}  Z={z_now:+.2f}  状态={'多价差' if pos_now==1 else '空价差' if pos_now==-1 else '空仓'}")
        print(f"   → {act}   ({why})")
        print(f"   建议手数: A({pa})×{la*units}  B({pb})×{lb*units}   约占保证金 ¥{margin*units:,.0f}")
    print("\n说明: 收盘后运行；'开仓'信号在次日开盘或当日尾盘按对冲手数建仓；")
    print("      止盈/止损按 Z 阈值；主力换月时同步移仓两条腿。实盘建议直接下券商挂牌套利指令。")


if __name__ == "__main__":
    main()
