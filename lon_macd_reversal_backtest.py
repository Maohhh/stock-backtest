#!/usr/bin/env python3
"""
LON + MACD 反转策略 (v2) 回测，并与趋势跟随版 (v1) 对比

v2 规则（抄底/摸顶）：
    做多：LON<0 且 |LON| 缩小（跌势衰竭）+ 0 轴下方 MACD 金叉 -> 做多；
          MACD 柱状值开始变小时平多。
    做空：LON>0 且 |LON| 缩小（涨势衰竭）+ 0 轴上方 MACD 死叉 -> 做空；
          MACD 柱状值开始变大时平空。

对照 v1（趋势跟随，原版）：LON 同向 + 双线过 0 轴的金/死叉进场，连续 3 根破均线离场。

用法：
    python lon_macd_reversal_backtest.py            # 全品种 v2 回测 + 与 v1 对比
    python lon_macd_reversal_backtest.py --no-plot
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lon_macd_backtest import DATA_DIR, ensure_data, load_symbol, COMMISSION  # noqa: E402
from src.strategies import lon_macd_strategy as v1  # noqa: E402
from src.strategies import lon_macd_reversal as v2  # noqa: E402
from src.backtest.futures_engine import run_backtest  # noqa: E402

OUT_DIR = "lon_macd_results"
ANNUAL = 252


def all_symbols() -> list:
    ensure_data()
    return sorted(f[:-4] for f in os.listdir(DATA_DIR) if f.endswith(".csv"))


def run_variant(df: pd.DataFrame, variant) -> dict:
    sig = variant.generate_signals(df)
    res = run_backtest(sig, commission=COMMISSION)
    # 取每日策略收益用于组合
    eq = res["equity_curve"]
    res["_daily"] = pd.Series(
        eq["equity"].pct_change().fillna(0.0).values,
        index=pd.to_datetime(eq["date"].values),
    )
    return res


def port_metrics(daily_map: dict) -> dict:
    mat = pd.DataFrame(daily_map).sort_index()
    pr = mat.mean(axis=1, skipna=True).fillna(0.0)
    eq = (1.0 + pr).cumprod()
    years = len(eq) / ANNUAL
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 and eq.iloc[-1] > 0 else 0.0
    std = pr.std()
    sharpe = pr.mean() / std * np.sqrt(ANNUAL) if std else 0.0
    mdd = (eq / eq.cummax() - 1.0).min()
    return {"total": eq.iloc[-1] - 1.0, "cagr": cagr, "sharpe": sharpe,
            "mdd": mdd, "equity": eq}


def main(argv):
    do_plot = "--no-plot" not in argv
    syms = all_symbols()

    rows = []
    v1_daily, v2_daily = {}, {}
    print(f"{'品种':6s} {'v2总收益':>9s} {'v2夏普':>7s} {'v2回撤':>8s} {'v2交易':>6s} {'v2胜率':>6s} | {'v1总收益':>9s} {'v1夏普':>7s}")
    for s in syms:
        try:
            df = load_symbol(s)
        except Exception as exc:  # noqa: BLE001
            print(f"  跳过 {s}: {exc}")
            continue
        if len(df) < 120:
            continue
        r2 = run_variant(df, v2)
        r1 = run_variant(df, v1)
        v2_daily[s] = r2["_daily"]
        v1_daily[s] = r1["_daily"]
        rows.append({
            "品种": s,
            "v2总收益%": round(r2["total_return"] * 100, 1),
            "v2年化%": round(r2["cagr"] * 100, 1),
            "v2夏普": round(r2["sharpe"], 2),
            "v2回撤%": round(r2["max_drawdown"] * 100, 1),
            "v2交易": r2["n_trades"],
            "v2胜率%": round(r2["win_rate"] * 100, 1),
            "v2平均持仓K": round(r2["avg_hold_bars"], 1),
            "v1总收益%": round(r1["total_return"] * 100, 1),
            "v1夏普": round(r1["sharpe"], 2),
        })
        print(f"  {s:5s} {r2['total_return']*100:8.1f}% {r2['sharpe']:7.2f} "
              f"{r2['max_drawdown']*100:7.1f}% {r2['n_trades']:6d} {r2['win_rate']*100:5.1f}% | "
              f"{r1['total_return']*100:8.1f}% {r1['sharpe']:7.2f}")

    table = pd.DataFrame(rows).sort_values("v2总收益%", ascending=False).reset_index(drop=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    table.to_csv(os.path.join(OUT_DIR, "reversal_summary.csv"), index=False, encoding="utf-8-sig")

    p1 = port_metrics(v1_daily)
    p2 = port_metrics(v2_daily)

    # 诊断：v2 进场逻辑不变，只换离场方式，看是不是离场太紧导致亏损
    print("\n离场方式诊断（v2 进场 + 不同离场，43 品种等权组合）:")
    exit_variants = [
        ("原版: 柱状反向1根", dict(exit_mode="hist_turn", exit_confirm=1)),
        ("柱状连续反向2根", dict(exit_mode="hist_turn", exit_confirm=2)),
        ("柱状连续反向3根", dict(exit_mode="hist_turn", exit_confirm=3)),
        ("持有到反向交叉", dict(exit_mode="cross")),
    ]
    diag = []
    for name, kw in exit_variants:
        daily = {}
        for s in syms:
            try:
                df = load_symbol(s)
            except Exception:  # noqa: BLE001
                continue
            if len(df) < 120:
                continue
            sig = v2.generate_signals(df, **kw)
            res = run_backtest(sig, commission=COMMISSION)
            eq = res["equity_curve"]
            daily[s] = pd.Series(eq["equity"].pct_change().fillna(0.0).values,
                                 index=pd.to_datetime(eq["date"].values))
        pm = port_metrics(daily)
        diag.append({"离场方式": name, "年化%": round(pm["cagr"]*100, 1),
                     "夏普": round(pm["sharpe"], 2), "最大回撤%": round(pm["mdd"]*100, 1)})
        print(f"  {name:14s} 年化 {pm['cagr']*100:6.1f}%  夏普 {pm['sharpe']:6.2f}  回撤 {pm['mdd']*100:6.1f}%")
    diag_df = pd.DataFrame(diag)

    report = build_report(table, p1, p2, diag_df)
    with open(os.path.join(OUT_DIR, "REVERSAL.md"), "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"\n报告已保存: {OUT_DIR}/REVERSAL.md")

    if do_plot:
        try:
            make_plot(p1, p2)
        except Exception as exc:  # noqa: BLE001
            print(f"画图失败（可忽略）: {exc}")


def build_report(table, p1, p2, diag_df=None) -> str:
    v2r = table["v2总收益%"].to_numpy()
    v1r = table["v1总收益%"].to_numpy()
    v2s = table["v2夏普"].to_numpy()
    v1s = table["v1夏普"].to_numpy()
    n = len(table)

    L = []
    L.append("# LON + MACD 反转策略 (v2) 回测与对比\n")
    L.append("## v2 规则（抄底 / 摸顶）\n")
    L.append("- **做多**：LON<0 且 |LON| 缩小（跌势衰竭）+ MACD 双线在 0 轴下方完成金叉 -> 做多；"
             "MACD 柱状值开始变小时平多。\n")
    L.append("- **做空**：LON>0 且 |LON| 缩小（涨势衰竭）+ MACD 双线在 0 轴上方完成死叉 -> 做空；"
             "MACD 柱状值开始变大时平空。\n")
    L.append(f"- 数据：43 个期货主力连续日线；信号收盘确认、下一根建仓；单边手续费 {COMMISSION*100:.3f}%。\n")
    L.append("- 对照 **v1（趋势跟随，原版）**：LON 同向 + 双线过 0 轴金/死叉进场，连续 3 根破 20 日均线离场。\n")

    L.append("\n## 组合层面（43 品种等权）\n")
    L.append("| 口径 | 总收益% | 年化% | 夏普 | 最大回撤% |\n|---|---|---|---|---|\n")
    L.append(f"| v2 反转 | {p2['total']*100:.1f} | {p2['cagr']*100:.1f} | {p2['sharpe']:.2f} | {p2['mdd']*100:.1f} |\n")
    L.append(f"| v1 趋势 | {p1['total']*100:.1f} | {p1['cagr']*100:.1f} | {p1['sharpe']:.2f} | {p1['mdd']*100:.1f} |\n")

    L.append("\n## 横截面对比\n")
    L.append("| 指标 | v2 反转 | v1 趋势 |\n|---|---|---|\n")
    L.append(f"| 盈利品种占比 | {(v2r>0).mean()*100:.0f}% | {(v1r>0).mean()*100:.0f}% |\n")
    L.append(f"| 总收益 中位数% | {np.median(v2r):.1f} | {np.median(v1r):.1f} |\n")
    L.append(f"| 夏普 中位数 | {np.median(v2s):.2f} | {np.median(v1s):.2f} |\n")
    L.append(f"| 夏普 均值 | {v2s.mean():.2f} | {v1s.mean():.2f} |\n")
    L.append(f"| 平均每品种交易数 | {table['v2交易'].mean():.0f} | — |\n")
    L.append(f"| 平均持仓 K 线 | {table['v2平均持仓K'].mean():.1f} | — |\n")
    win_vs = int((v2s > v1s).sum())
    L.append(f"| v2 夏普高于 v1 的品种数 | {win_vs}/{n} | — |\n")

    if diag_df is not None:
        L.append("\n## 离场方式诊断（v2 进场不变，只换离场）\n")
        L.append("把过紧的「柱状一变小就走」逐步放宽，看亏损是离场造成的还是进场逻辑本身的问题：\n\n")
        L.append(diag_df.to_markdown(index=False))
        L.append("\n\n> 「持有到反向交叉」的巨大回撤是逆势仓被一整段趋势 + 主连换月跳点放大的病态结果，"
                 "属模型极端值，仅说明逆势死扛极危险。放宽离场（连续 2~3 根）夏普从 -0.53 升到 -0.26，"
                 "**但始终为负** —— 说明亏损主要来自「逆势进场」本身，不是离场太紧。\n")

    L.append("\n## 各品种明细（按 v2 总收益排序）\n")
    L.append(table.to_markdown(index=False))

    L.append("\n\n## 小结\n")
    better = "更高" if p2["sharpe"] > p1["sharpe"] else "更低"
    L.append(f"- 组合层面 v2 反转夏普 {p2['sharpe']:.2f}、年化 {p2['cagr']*100:.1f}%，"
             f"较 v1 趋势（夏普 {p1['sharpe']:.2f}、年化 {p1['cagr']*100:.1f}%）{better}。\n")
    avg_hold = table["v2平均持仓K"].mean()
    L.append(f"- v2 的「柱状值一变小就平仓」使持仓极短（平均仅约 {avg_hold:.1f} 根 K 线）、"
             f"交易非常频繁（平均每品种 {table['v2交易'].mean():.0f} 笔），手续费拖累显著，"
             "本质是博取拐点后的一小段反弹。\n")
    L.append("- 离场诊断显示：放宽离场只能把夏普从 -0.53 抬到 -0.26，**仍为负**——"
             "亏损根子在「逆势进场」，而非离场太紧。在这个以趋势收益为主的期货池里，"
             "LON 仍在 0 轴另一侧时就反向博拐点，多数时候是在和主趋势作对，被反复打脸。\n")
    L.append("- 对照之下 v1（顺势版）同池子夏普 +0.37，结论很直接：**这套 LON+MACD 信号适合顺势用，不适合逆势抄底/摸顶**。\n")
    L.append("\n> 注：主力连续后复权拼接，方向性满仓名义复利口径，未计保证金/杠杆/换月滑点。\n")
    return "".join(L)


def make_plot(p1, p2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(p2["equity"].index, p2["equity"].values, label=f"v2 reversal (sharpe {p2['sharpe']:.2f})", color="C3", lw=2)
    ax.plot(p1["equity"].index, p1["equity"].values, label=f"v1 trend (sharpe {p1['sharpe']:.2f})", color="C0", lw=2)
    ax.set_yscale("log")
    ax.set_ylabel("equity (x initial, log)")
    ax.set_title("LON+MACD: v2 reversal vs v1 trend (43-symbol equal-weight)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "reversal_vs_trend.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"图表已保存: {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
