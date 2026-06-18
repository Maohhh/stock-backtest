#!/usr/bin/env python3
"""
LON + MACD 趋势跟随策略 —— 期货主力连续日线回测

策略：
    多头：LON > 0（上涨趋势）+ MACD 金叉且 DIF/DEA 均在 0 轴上方 -> 开多；
          连续 3 根 K 线收盘跌破 20 日均线 -> 平多。
    空头：LON < 0（下跌趋势）+ MACD 死叉且 DIF/DEA 均在 0 轴下方 -> 开空；
          连续 3 根 K 线收盘站上 20 日均线 -> 平空。

数据：data_futures/sina_daily_main/*.csv（新浪期货主力连续后复权拼接日线）。
若本地缺失，会尝试从数据分支 origin/claude/futures-data-inventory-fdwz31 提取。

用法：
    python lon_macd_backtest.py                 # 回测全部品种
    python lon_macd_backtest.py RB0 IF0 CU0     # 只测指定品种
    python lon_macd_backtest.py --no-plot       # 跳过画图
"""

import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.strategies.lon_macd_strategy import generate_signals  # noqa: E402
from src.backtest.futures_engine import run_backtest  # noqa: E402

DATA_DIR = os.path.join("data_futures", "sina_daily_main")
DATA_REF = "origin/claude/futures-data-inventory-fdwz31"
OUT_DIR = "lon_macd_results"
COMMISSION = 0.0003  # 单边万分之三


def ensure_data() -> None:
    """本地缺数据时，从数据分支提取 sina_daily_main 到工作目录。"""
    if os.path.isdir(DATA_DIR) and any(f.endswith(".csv") for f in os.listdir(DATA_DIR)):
        return
    print(f"本地无数据，尝试从 {DATA_REF} 提取 ...")
    os.makedirs(DATA_DIR, exist_ok=True)
    files = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", DATA_REF]
    ).decode().splitlines()
    files = [f for f in files if f.startswith(DATA_DIR) and f.endswith(".csv")]
    for f in files:
        content = subprocess.check_output(["git", "show", f"{DATA_REF}:{f}"])
        with open(f, "wb") as fh:
            fh.write(content)
    print(f"已提取 {len(files)} 个文件。")


def load_symbol(symbol: str) -> pd.DataFrame:
    """读取单个主力连续 CSV，标准化为 date/open/high/low/close/volume。"""
    path = os.path.join(DATA_DIR, f"{symbol}.csv")
    raw = pd.read_csv(path)
    df = raw.rename(
        columns={"d": "date", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    )
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    # 去掉首段无效行（部分品种早期含 0 价）
    df = df[df["close"] > 0].reset_index(drop=True)
    return df


def run_symbol(symbol: str) -> dict:
    df = load_symbol(symbol)
    if len(df) < 120:
        return None
    signal_df = generate_signals(df)
    result = run_backtest(signal_df, commission=COMMISSION)
    result["symbol"] = symbol
    result["start"] = df["date"].iloc[0].date()
    result["end"] = df["date"].iloc[-1].date()
    return result


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def main(argv: list) -> None:
    do_plot = "--no-plot" not in argv
    symbols = [a for a in argv if not a.startswith("--")]

    ensure_data()
    if not symbols:
        symbols = sorted(
            f[:-4] for f in os.listdir(DATA_DIR) if f.endswith(".csv")
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    for sym in symbols:
        try:
            res = run_symbol(sym)
        except Exception as exc:  # noqa: BLE001
            print(f"  跳过 {sym}: {exc}")
            continue
        if res is None:
            continue
        results.append(res)
        print(
            f"  {sym:5s} {res['start']}~{res['end']}  "
            f"总收益 {pct(res['total_return']):>8s}  "
            f"年化 {pct(res['cagr']):>7s}  "
            f"夏普 {res['sharpe']:5.2f}  "
            f"回撤 {pct(res['max_drawdown']):>7s}  "
            f"交易 {res['n_trades']:3d}  "
            f"胜率 {pct(res['win_rate']):>6s}"
        )

    if not results:
        print("没有可回测的品种。")
        return

    summary = build_summary(results)
    summary_path = os.path.join(OUT_DIR, "summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n汇总已保存: {summary_path}")

    report = build_report(results, summary)
    report_path = os.path.join(OUT_DIR, "REPORT.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"报告已保存: {report_path}")

    # 保存逐笔交易（合并）
    all_trades = []
    for res in results:
        t = res["trades"].copy()
        if len(t):
            t.insert(0, "symbol", res["symbol"])
            all_trades.append(t)
    if all_trades:
        trades_df = pd.concat(all_trades, ignore_index=True)
        trades_path = os.path.join(OUT_DIR, "trades.csv")
        trades_df.to_csv(trades_path, index=False, encoding="utf-8-sig")
        print(f"逐笔交易已保存: {trades_path}")

    if do_plot:
        try:
            make_plots(results)
        except Exception as exc:  # noqa: BLE001
            print(f"画图失败（可忽略）: {exc}")


def build_summary(results: list) -> pd.DataFrame:
    rows = []
    for res in results:
        rows.append(
            {
                "品种": res["symbol"],
                "起始": res["start"],
                "结束": res["end"],
                "总收益%": round(res["total_return"] * 100, 1),
                "年化%": round(res["cagr"] * 100, 1),
                "买入持有%": round(res["bh_return"] * 100, 1),
                "夏普": round(res["sharpe"], 2),
                "索提诺": round(res["sortino"], 2),
                "卡玛": round(res["calmar"], 2),
                "最大回撤%": round(res["max_drawdown"] * 100, 1),
                "交易次数": res["n_trades"],
                "多头次数": res["long_trades"],
                "空头次数": res["short_trades"],
                "胜率%": round(res["win_rate"] * 100, 1),
                "盈亏比": round(res["profit_factor"], 2) if res["profit_factor"] != float("inf") else 999.0,
                "平均持仓K": round(res["avg_hold_bars"], 1),
                "持仓占比%": round(res["exposure"] * 100, 1),
            }
        )
    df = pd.DataFrame(rows).sort_values("总收益%", ascending=False).reset_index(drop=True)
    return df


def build_report(results: list, summary: pd.DataFrame) -> str:
    import numpy as np

    n = len(results)
    rets = np.array([r["total_return"] for r in results])
    cagrs = np.array([r["cagr"] for r in results])
    sharpes = np.array([r["sharpe"] for r in results])
    mdds = np.array([r["max_drawdown"] for r in results])
    win = (rets > 0).sum()
    total_trades = sum(r["n_trades"] for r in results)
    all_win_rates = np.array([r["win_rate"] for r in results])

    lines = []
    lines.append("# LON + MACD 趋势跟随策略回测报告\n")
    lines.append("## 策略规则\n")
    lines.append(
        "- **多头**：LON 长线指标 > 0（上涨趋势）且 MACD 金叉、DIF/DEA 均 > 0 时开多；"
        "连续 3 根 K 线收盘价跌破 20 日均线时平多。\n"
        "- **空头**：LON 长线指标 < 0（下跌趋势）且 MACD 死叉、DIF/DEA 均 < 0 时开空；"
        "连续 3 根 K 线收盘价站上 20 日均线时平空。\n"
    )
    lines.append("## 回测设置\n")
    lines.append(
        f"- 数据：新浪期货**主力连续后复权拼接日线**，共 **{n}** 个品种。\n"
        f"- 信号收盘确认、下一根 K 线建仓（无未来函数）；满仓名义本金、不加杠杆复利计收益。\n"
        f"- 单边手续费：{COMMISSION * 100:.3f}%（换仓扣费，反手按两次单边）。\n"
        f"- MACD(12,26,9)、MA20、LON(10,20)、连续 3 根离场、预热 60 根。\n"
    )
    lines.append("## 组合层面汇总\n")
    lines.append(
        f"| 指标 | 数值 |\n|---|---|\n"
        f"| 品种数 | {n} |\n"
        f"| 盈利品种 / 占比 | {win} / {win / n * 100:.0f}% |\n"
        f"| 总收益 中位数 | {np.median(rets) * 100:.1f}% |\n"
        f"| 总收益 平均 | {rets.mean() * 100:.1f}% |\n"
        f"| 年化 中位数 | {np.median(cagrs) * 100:.1f}% |\n"
        f"| 夏普 中位数 | {np.median(sharpes):.2f} |\n"
        f"| 夏普 平均 | {sharpes.mean():.2f} |\n"
        f"| 最大回撤 中位数 | {np.median(mdds) * 100:.1f}% |\n"
        f"| 单笔胜率 中位数 | {np.median(all_win_rates) * 100:.1f}% |\n"
        f"| 总交易笔数 | {total_trades} |\n"
    )

    lines.append("\n## 各品种明细（按总收益排序）\n")
    lines.append(summary.to_markdown(index=False))
    lines.append("\n\n## 说明与局限\n")
    lines.append(
        "- 主力连续为后复权拼接序列，已尽量消除换月跳空，但仍非真实可交易合约价格；"
        "实盘需考虑换月滑点、保证金与流动性。\n"
        "- 收益以方向性百分比复利衡量（满仓名义、不加杠杆），便于跨品种横向比较，"
        "不等同于按手数 / 保证金的真实资金曲线。\n"
        "- 未计入隔夜资金成本、冲击成本；手续费用统一比例近似。\n"
    )
    return "\n".join(lines)


def make_plots(results: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) 各品种净值曲线总览
    ranked = sorted(results, key=lambda r: r["total_return"], reverse=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    for res in ranked:
        eq = res["equity_curve"]
        ax.plot(eq["date"], eq["equity"], linewidth=0.9, alpha=0.8)
    ax.set_title("LON+MACD strategy equity curves (all symbols)")
    ax.set_ylabel("equity (x initial)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "equity_all.png"), dpi=120)
    plt.close(fig)

    # 2) Top/Bottom 品种 strategy vs buy&hold
    show = ranked[:4] + ranked[-2:]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    for ax, res in zip(axes.flat, show):
        eq = res["equity_curve"]
        ax.plot(eq["date"], eq["equity"], label="strategy", color="C0")
        ax.plot(eq["date"], eq["bh_equity"], label="buy&hold", color="C1", alpha=0.7)
        ax.set_title(f"{res['symbol']}  ret={res['total_return']*100:.0f}%  sharpe={res['sharpe']:.2f}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Strategy vs Buy&Hold (top 4 / bottom 2)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "top_bottom.png"), dpi=120)
    plt.close(fig)
    print(f"图表已保存: {OUT_DIR}/equity_all.png, {OUT_DIR}/top_bottom.png")


if __name__ == "__main__":
    main(sys.argv[1:])
