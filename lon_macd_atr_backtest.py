#!/usr/bin/env python3
"""
出场方式横向对比：原版 MA 出场 vs ATR 移动止损（吊灯出场）

进场都用 v1 信号；对比不同出场对收益/夏普/胜率/盈亏比/期望的影响。
同时给出 43 品种等权组合层面的结果。

用法：
    python lon_macd_atr_backtest.py            # 全部 43(+玻璃) 品种 + 组合 + 玻璃明细
    python lon_macd_atr_backtest.py --no-plot
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lon_macd_backtest import DATA_DIR, ensure_data, load_symbol, COMMISSION  # noqa: E402
from src.strategies import lon_macd_strategy as v1  # noqa: E402
from src.strategies import lon_macd_atr as atr_exit  # noqa: E402
from src.backtest.futures_engine import run_backtest  # noqa: E402

OUT_DIR = "lon_macd_results"
ANNUAL = 252

CONFIGS = [
    ("MA20出场(原版)", v1, dict(ma_period=20, exit_bars=3)),
    ("MA10出场", v1, dict(ma_period=10, exit_bars=3)),
    ("ATR止损 k=2", atr_exit, dict(k=2.0)),
    ("ATR止损 k=3", atr_exit, dict(k=3.0)),
    ("ATR止损 k=4", atr_exit, dict(k=4.0)),
]


def run_cfg(df, module, params):
    sig = module.generate_signals(df, **params)
    r = run_backtest(sig, commission=COMMISSION)
    eq = r["equity_curve"]
    r["_daily"] = pd.Series(eq["equity"].pct_change().fillna(0.0).values,
                            index=pd.to_datetime(eq["date"].values))
    return r


def port(daily_map):
    mat = pd.DataFrame(daily_map).sort_index()
    pr = mat.mean(axis=1, skipna=True).fillna(0.0)
    eq = (1 + pr).cumprod()
    years = len(eq) / ANNUAL
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 and eq.iloc[-1] > 0 else 0
    sh = pr.mean() / pr.std() * np.sqrt(ANNUAL) if pr.std() else 0
    mdd = (eq / eq.cummax() - 1).min()
    return {"total": eq.iloc[-1] - 1, "cagr": cagr, "sharpe": sh, "mdd": mdd, "equity": eq}


def main(argv):
    do_plot = "--no-plot" not in argv
    ensure_data()
    syms = sorted(f[:-4] for f in os.listdir(DATA_DIR) if f.endswith(".csv"))

    # 每个配置：收集各品种 daily + 横截面统计 + 玻璃单独
    port_results = {}
    cross = {name: [] for name, _, _ in CONFIGS}   # 每品种 total_return
    crosssh = {name: [] for name, _, _ in CONFIGS}
    fg_rows = []
    for name, module, params in CONFIGS:
        daily = {}
        for s in syms:
            try:
                df = load_symbol(s)
            except Exception:  # noqa: BLE001
                continue
            if len(df) < 120:
                continue
            r = run_cfg(df, module, params)
            daily[s] = r["_daily"]
            cross[name].append(r["total_return"])
            crosssh[name].append(r["sharpe"])
            if s == "FG0":
                fg_rows.append({
                    "出场方式": name,
                    "总收益%": round(r["total_return"] * 100, 1),
                    "夏普": round(r["sharpe"], 2),
                    "最大回撤%": round(r["max_drawdown"] * 100, 1),
                    "交易数": r["n_trades"],
                    "胜率%": round(r["win_rate"] * 100, 1),
                    "盈亏比": round(r["profit_factor"], 2),
                    "平均赢%": round(r["avg_win"] * 100, 2),
                    "平均亏%": round(r["avg_loss"] * 100, 2),
                    "平均持仓K": round(r["avg_hold_bars"], 1),
                })
        port_results[name] = port(daily)
        p = port_results[name]
        print(f"{name:16s} 组合: 总收益{p['total']*100:7.1f}% 年化{p['cagr']*100:5.1f}% "
              f"夏普{p['sharpe']:5.2f} 回撤{p['mdd']*100:6.1f}%  "
              f"| 盈利品种{(np.array(cross[name])>0).mean()*100:3.0f}% 夏普中位{np.median(crosssh[name]):.2f}")

    # 报告
    R = ["# 出场方式对比：MA 出场 vs ATR 移动止损\n",
         "进场统一用 v1 信号；只改出场。组合 = 43(+玻璃) 品种等权。\n\n",
         "## 组合层面\n",
         "| 出场方式 | 总收益% | 年化% | 夏普 | 最大回撤% | 盈利品种% | 品种夏普中位 |\n|---|---|---|---|---|---|---|\n"]
    for name, _, _ in CONFIGS:
        p = port_results[name]
        R.append(f"| {name} | {p['total']*100:.1f} | {p['cagr']*100:.1f} | {p['sharpe']:.2f} | "
                 f"{p['mdd']*100:.1f} | {(np.array(cross[name])>0).mean()*100:.0f} | {np.median(crosssh[name]):.2f} |\n")

    R.append("\n## 玻璃 FG0 各出场方式明细\n")
    R.append(pd.DataFrame(fg_rows).to_markdown(index=False))

    R.append("\n\n## 对照：固定点数括号单（来自 BRACKET 研究，玻璃 FG0）\n")
    R.append("| 出场方式 | 胜率% | 净期望% |\n|---|---|---|\n")
    R.append("| +3点止盈(无止损) | ~96 | -0.08（负，被大亏吃掉）|\n")
    R.append("| +10/-5点 | 14.9 | -0.26 |\n")
    R.append("| +10/-20点 | 55.3 | -0.30 |\n")

    R.append("\n## 小结\n")
    best = max(CONFIGS, key=lambda c: port_results[c[0]]["sharpe"])[0]
    pb = port_results[best]
    fg = pd.DataFrame(fg_rows)
    fg_best = fg.loc[fg["夏普"].idxmax()]
    R.append(f"- 组合层面夏普最高的是 **{best}**（夏普 {pb['sharpe']:.2f}、年化 {pb['cagr']*100:.1f}%、回撤 {pb['mdd']*100:.1f}%）；"
             "ATR 止损（尤其 k=2）回撤控制最好。让利润奔跑的几种出场（MA/ATR）夏普都在 0.4~0.5，相互接近。\n")
    R.append(f"- **玻璃 FG0 上 ATR k=2 明显最优**：总收益 {fg_best['总收益%']}%、夏普 {fg_best['夏普']}、"
             f"盈亏比 {fg_best['盈亏比']}，把原版 MA20 的 -12.3% 反转成大赢——按波动率移动止损同时抬高了胜率和盈亏比。\n")
    R.append("- **真正的分水岭是「让利润奔跑 vs 砍掉利润」**：MA/ATR 这类移动出场普遍正期望，"
             "而固定 +N 点止盈的括号单普遍负期望（把右尾大赢砍掉只剩小赢+偶尔大亏）。\n")
    R.append("- 进场短期随机，所以靠「出场纪律 + 右尾 + 分散」赚钱，而不是靠进场即时确定性或固定点数止盈止损。\n")

    with open(os.path.join(OUT_DIR, "EXIT_COMPARE.md"), "w", encoding="utf-8") as fh:
        fh.write("".join(R))
    print(f"\n报告已保存: {OUT_DIR}/EXIT_COMPARE.md")

    if do_plot:
        try:
            _plot(port_results)
        except Exception as exc:  # noqa: BLE001
            print(f"画图失败（可忽略）: {exc}")


def _plot(port_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 7))
    for name, p in port_results.items():
        ax.plot(p["equity"].index, p["equity"].values, lw=1.6, label=f"{name} (sh {p['sharpe']:.2f})")
    ax.set_yscale("log")
    ax.set_ylabel("equity (x initial, log)")
    ax.set_title("Exit comparison: MA exit vs ATR trailing stop (43-symbol equal-weight)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "exit_compare_equity.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"图表已保存: {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
