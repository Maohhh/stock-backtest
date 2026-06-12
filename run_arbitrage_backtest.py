"""
期货价差套利回测

使用仓库的日线数据获取能力（新浪期货接口）回测以下套利品种：

  1. 菜粕 9-11      RM09-RM11    价差 < 50  做多，回归到本周期均值平仓
  2. 鸡蛋 9-10      JD09-JD10    价差 < 100 做多，回归到本周期均值平仓
  3. 花生 3-4       PK03-PK04    > 70 做空 / < -70 做多，回归到 0 平仓
  4. PTA/短纤       TA-PF(主连)  > -1400 做空，回归到中枢(滚动均值)平仓
  5. 玉米/玉米淀粉  C-CS(主连)   > -300  做空，回归到中枢(滚动均值)平仓

跨期价差「交割月不计入」：在近月合约交割月前最后一个交易日强制平仓。

运行：
    python run_arbitrage_backtest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.arbitrage.sina_futures import clean_spikes, fetch_close
from src.arbitrage.spread_backtest import SpreadResult, backtest_spread


# ---------------------------------------------------------------------------
# 跨期价差：按年构造合约对，剔除近月交割月
# ---------------------------------------------------------------------------
def calendar_spread(front_sym: str, back_sym: str, delivery_month: int) -> pd.Series:
    """构造单个周期的跨期价差，剔除近月交割月当月及之后的数据。

    front_sym 如 ``RM2509``，从中解析交割年份；delivery_month 为近月交割月。
    """
    year = 2000 + int(front_sym[-4:-2])
    cutoff = pd.Timestamp(year=year, month=delivery_month, day=1)

    f = fetch_close(front_sym)
    b = fetch_close(back_sym)
    if f.empty or b.empty:
        return pd.Series(dtype=float)
    spread = (f - b).dropna()
    spread = spread[spread.index < cutoff]
    spread.name = f"{front_sym}-{back_sym}"
    return spread


def run_calendar(
    name: str,
    prefix: str,
    front_mm: str,
    back_mm: str,
    years: list[int],
    delivery_month: int,
    *,
    entry_long=None,
    entry_short=None,
    exit_mode: str,
    multiplier: float,
    slippage: float,
    upper: bool = True,
) -> SpreadResult:
    """逐年回测跨期价差，汇总所有周期的交易。

    prefix 大小写需匹配交易所：郑商所大写(RM/PK)，大商所小写(jd)。
    """
    merged = SpreadResult(name=name, multiplier=multiplier)
    equities = []
    for y in years:
        yy = f"{y % 100:02d}"
        front = f"{prefix}{yy}{front_mm}"
        back = f"{prefix}{yy}{back_mm}"
        try:
            spread = calendar_spread(front, back, delivery_month)
        except Exception as exc:  # noqa: BLE001
            print(f"  [跳过] {front}-{back}: {exc}")
            continue
        if spread.empty or len(spread) < 30:
            continue
        res = backtest_spread(
            spread,
            name=f"{name}{y}",
            entry_long=entry_long,
            entry_short=entry_short,
            exit_mode=exit_mode,
            force_close=True,
            slippage=slippage,
            multiplier=multiplier,
        )
        merged.trades.extend(res.trades)
        equities.append(res.equity)
        lo, hi, last = spread.min(), spread.max(), spread.iloc[-1]
        print(
            f"  {y} {front}-{back:8s} n={len(spread):3d} "
            f"价差[{lo:7.1f},{hi:7.1f}] 末={last:7.1f} "
            f"交易={res.n_trades} 点数={res.total_points:7.1f}"
        )
    # 拼接各周期权益（点数累计）
    if equities:
        acc = 0.0
        parts = []
        for eq in equities:
            parts.append(eq + acc)
            acc += eq.iloc[-1] if len(eq) else 0.0
        merged.equity = pd.concat(parts)
    return merged


def run_cross(
    name: str,
    sym_a: str,
    sym_b: str,
    *,
    entry_short: float,
    rolling_window: int,
    multiplier: float,
    slippage: float,
    start: str | None = None,
) -> SpreadResult:
    """跨品种价差（主力连续）：> entry_short 做空，回归到滚动均值中枢平仓。"""
    a = fetch_close(sym_a, use_cache=False)
    b = fetch_close(sym_b, use_cache=False)
    spread = (a - b).dropna()
    spread = clean_spikes(spread, max_jump=500)  # 剔除主力连续拼接尖刺
    if start:
        spread = spread[spread.index >= start]
    spread.name = f"{sym_a}-{sym_b}"
    lo, hi, mean, last = spread.min(), spread.max(), spread.mean(), spread.iloc[-1]
    print(
        f"  {sym_a}-{sym_b} n={len(spread)} "
        f"价差[{lo:8.1f},{hi:8.1f}] 均值={mean:8.1f} 末={last:8.1f}"
    )
    return backtest_spread(
        spread,
        name=name,
        entry_short=entry_short,
        exit_mode="rolling_mean",
        rolling_window=rolling_window,
        force_close=False,
        slippage=slippage,
        multiplier=multiplier,
    )


# 中文品种名 -> ASCII 标签（避免无中文字体时图中显示方块）
_ASCII = {
    "菜粕9-11": "RM 9-11",
    "鸡蛋9-10": "JD 9-10",
    "花生3-4": "PK 3-4",
    "PTA/短纤": "TA-PF",
    "玉米/淀粉": "C-CS",
}


def plot_equity(results: list[SpreadResult], path: Path) -> None:
    """绘制各品种累计价差点数权益曲线 + 总点数柱状图。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[跳过绘图] {exc}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    for r in results:
        if not r.equity.empty:
            ax1.plot(r.equity.index, r.equity.values, label=_ASCII.get(r.name, r.name))
    ax1.axhline(0, color="gray", lw=0.8, ls="--")
    ax1.set_title("Cumulative spread PnL (points)")
    ax1.set_ylabel("points")
    ax1.legend()
    ax1.grid(alpha=0.3)

    labels = [_ASCII.get(r.name, r.name) for r in results]
    totals = [r.total_points for r in results]
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in totals]
    ax2.bar(labels, totals, color=colors)
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_title("Total spread points by product")
    ax2.set_ylabel("points")
    for i, v in enumerate(totals):
        ax2.text(i, v, f"{v:.0f}", ha="center", va="bottom" if v >= 0 else "top")
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"权益曲线已保存至 {path}")


def report(results: list[SpreadResult]) -> None:
    print("\n" + "=" * 96)
    print("套利回测汇总（价差点数为毛利，元/手对未计手续费滑点）")
    print("=" * 96)
    header = (
        f"{'品种':<14}{'交易数':>6}{'胜率':>8}{'总点数':>10}"
        f"{'均点/笔':>9}{'最好':>8}{'最差':>8}{'持仓天':>7}{'最大回撤(点)':>13}{'≈元/手对':>12}"
    )
    print(header)
    print("-" * 96)
    for r in results:
        print(
            f"{r.name:<14}{r.n_trades:>6}{r.win_rate*100:>7.1f}%"
            f"{r.total_points:>10.1f}{r.avg_points:>9.1f}{r.best:>8.1f}"
            f"{r.worst:>8.1f}{r.avg_hold_days:>7.0f}"
            f"{r.max_drawdown_points:>13.1f}{r.total_yuan:>12,.0f}"
        )
    print("-" * 96)


def main() -> None:
    results: list[SpreadResult] = []

    print("[1] 菜粕 9-11（RM09-RM11，<50 做多，回归本周期均值）")
    results.append(
        run_calendar(
            "菜粕9-11", "RM", "09", "11",
            years=list(range(2019, 2027)), delivery_month=9,
            entry_long=50, exit_mode="cycle_mean",
            multiplier=10, slippage=2,
        )
    )

    print("[2] 鸡蛋 9-10（JD09-JD10，<100 做多，回归本周期均值）")
    results.append(
        run_calendar(
            "鸡蛋9-10", "jd", "09", "10",
            years=list(range(2019, 2027)), delivery_month=9,
            entry_long=100, exit_mode="cycle_mean",
            multiplier=10, slippage=3,
        )
    )

    print("[3] 花生 3-4（PK03-PK04，>70 做空 / <-70 做多，回归 0）")
    results.append(
        run_calendar(
            "花生3-4", "PK", "03", "04",
            years=list(range(2022, 2028)), delivery_month=3,
            entry_long=-70, entry_short=70, exit_mode="zero",
            multiplier=5, slippage=2,
        )
    )

    print("[4] PTA/短纤（TA-PF 主力连续，>-1400 做空，回归滚动均值中枢）")
    results.append(
        run_cross(
            "PTA/短纤", "TA0", "PF0",
            entry_short=-1400, rolling_window=120,
            multiplier=5, slippage=4, start="2020-10-12",
        )
    )

    print("[5] 玉米/玉米淀粉（C-CS 主力连续，>-300 做空，回归滚动均值中枢）")
    results.append(
        run_cross(
            "玉米/淀粉", "c0", "cs0",
            entry_short=-300, rolling_window=120,
            multiplier=10, slippage=3, start="2016-01-01",
        )
    )

    report(results)

    # 导出交易明细
    out_dir = Path(__file__).resolve().parent / "data" / "arbitrage_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        df = r.to_frame()
        if not df.empty:
            fname = r.name.replace("/", "_") + ".csv"
            df.to_csv(out_dir / fname, index=False)
    print(f"\n交易明细已导出至 {out_dir}")

    plot_equity(results, out_dir / "equity_curves.png")


if __name__ == "__main__":
    main()
