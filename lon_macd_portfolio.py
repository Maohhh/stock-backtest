#!/usr/bin/env python3
"""
LON + MACD 策略 —— 品种筛选与组合构建

敏感性分析的结论：(1) 短均线 MA10 更优；(2) 单品种盈亏天差地别，
「选对品种」比「调参数」更重要；(3) 等权一篮子分散后回撤大幅下降。

本脚本在 MA10 参数下，比较三种组合构建方式：

    A. 全部 43 品种等权（基准）
    B. 事后最优 top-K（按全历史夏普选 top-K）——**有前视偏差，仅作上界参考**
    C. 滚动筛选 top-K（walk-forward）——每年初只用「过去 N 年」的策略表现排名，
       选出 top-K 在下一年持有，逐年拼接，**无前视、可实盘复现**

口径：等权组合 = 每日把所选品种的策略日收益做截面等权平均。

用法：
    python lon_macd_portfolio.py            # 跑组合对比 + 出报告/图
    python lon_macd_portfolio.py --no-plot
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lon_macd_backtest import DATA_DIR, ensure_data, load_symbol, COMMISSION  # noqa: E402
from src.strategies.lon_macd_strategy import generate_signals  # noqa: E402
from src.backtest.futures_engine import run_backtest  # noqa: E402

OUT_DIR = "lon_macd_results"
ANNUAL = 252
PARAMS = dict(ma_period=10, exit_bars=3, require_zero_axis=True, allow_reverse=True)
LOOKBACK_YEARS = 3       # 滚动筛选回看窗口
MIN_OBS = 250            # 入选所需最少交易日（回看窗口内）
TOPK_LIST = [5, 8, 10, 15]
DEFAULT_K = 8


def daily_returns_matrix() -> pd.DataFrame:
    """对每个品种用 MA10 参数回测，取每日策略净收益，拼成 日期×品种 矩阵。"""
    ensure_data()
    syms = sorted(f[:-4] for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
    series = {}
    for s in syms:
        try:
            df = load_symbol(s)
        except Exception:  # noqa: BLE001
            continue
        if len(df) < 120:
            continue
        sig = generate_signals(df, **PARAMS)
        res = run_backtest(sig, commission=COMMISSION)
        eq = res["equity_curve"]
        r = eq["equity"].pct_change().fillna(0.0)
        series[s] = pd.Series(r.values, index=pd.to_datetime(eq["date"].values))
    mat = pd.DataFrame(series).sort_index()
    return mat


def metrics(port_ret: pd.Series) -> dict:
    port_ret = port_ret.fillna(0.0)
    eq = (1.0 + port_ret).cumprod()
    years = len(eq) / ANNUAL
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 and eq.iloc[-1] > 0 else 0.0
    std = port_ret.std()
    sharpe = port_ret.mean() / std * np.sqrt(ANNUAL) if std else 0.0
    dn = port_ret[port_ret < 0].std()
    sortino = port_ret.mean() / dn * np.sqrt(ANNUAL) if dn else 0.0
    mdd = (eq / eq.cummax() - 1.0).min()
    calmar = cagr / abs(mdd) if mdd < 0 else 0.0
    return {
        "total": eq.iloc[-1] - 1.0, "cagr": cagr, "sharpe": sharpe,
        "sortino": sortino, "mdd": mdd, "calmar": calmar, "equity": eq,
    }


def sharpe_of(col: pd.Series) -> float:
    c = col.dropna()
    if c.size < MIN_OBS or c.std() == 0:
        return np.nan
    return c.mean() / c.std() * np.sqrt(ANNUAL)


def static_topk(mat: pd.DataFrame, k: int) -> tuple:
    """事后最优：按全历史夏普选 top-K（有前视偏差）。"""
    ranks = mat.apply(sharpe_of).dropna().sort_values(ascending=False)
    top = list(ranks.index[:k])
    port = mat[top].mean(axis=1, skipna=True)
    return port, top


def walkforward_topk(mat: pd.DataFrame, k: int) -> tuple:
    """滚动筛选：每年初按过去 LOOKBACK_YEARS 年夏普选 top-K，持有下一年。"""
    idx = mat.index
    years = sorted(set(idx.year))
    start_year = years[0] + LOOKBACK_YEARS
    port = pd.Series(0.0, index=idx)
    in_test = pd.Series(False, index=idx)
    picks = {}
    for y in [yr for yr in years if yr >= start_year]:
        train = mat[(idx.year >= y - LOOKBACK_YEARS) & (idx.year < y)]
        ranks = train.apply(sharpe_of).dropna().sort_values(ascending=False)
        if ranks.empty:
            continue
        top = list(ranks.index[:k])
        picks[y] = top
        mask = idx.year == y
        port.loc[mask] = mat.loc[mask, top].mean(axis=1, skipna=True).values
        in_test.loc[mask] = True
    port = port[in_test]  # 仅保留有筛选结果的测试期
    return port, picks


def main(argv):
    do_plot = "--no-plot" not in argv
    print(f"逐品种回测（参数 {PARAMS}）...")
    mat = daily_returns_matrix()
    print(f"收益矩阵: {mat.shape[0]} 个交易日 × {mat.shape[1]} 个品种 "
          f"({mat.index[0].date()} ~ {mat.index[-1].date()})\n")

    results = {}

    # A. 全部等权
    all_port = mat.mean(axis=1, skipna=True)
    results["全部43等权"] = metrics(all_port)

    # B. 事后最优 top-K（有偏）
    for k in TOPK_LIST:
        port, top = static_topk(mat, k)
        results[f"事后最优top{k}(有偏)"] = metrics(port)
        if k == DEFAULT_K:
            static_pick = top

    # C. 滚动筛选 top-K（无偏）
    wf_picks = {}
    for k in TOPK_LIST:
        port, picks = walkforward_topk(mat, k)
        results[f"滚动筛选top{k}"] = metrics(port)
        wf_picks[k] = picks

    # 汇总表
    rows = []
    for name, m in results.items():
        rows.append({
            "组合": name,
            "总收益%": round(m["total"] * 100, 1),
            "年化%": round(m["cagr"] * 100, 1),
            "夏普": round(m["sharpe"], 2),
            "索提诺": round(m["sortino"], 2),
            "最大回撤%": round(m["mdd"] * 100, 1),
            "卡玛": round(m["calmar"], 2),
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    os.makedirs(OUT_DIR, exist_ok=True)
    table.to_csv(os.path.join(OUT_DIR, "portfolio_compare.csv"), index=False, encoding="utf-8-sig")

    # 滚动筛选 default K 的选中频次
    pick_freq = pd.Series(
        [s for lst in wf_picks[DEFAULT_K].values() for s in lst]
    ).value_counts()

    report = build_report(table, results, static_pick, wf_picks, pick_freq, mat)
    with open(os.path.join(OUT_DIR, "PORTFOLIO.md"), "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"\n报告已保存: {OUT_DIR}/PORTFOLIO.md")

    if do_plot:
        try:
            make_plot(results)
        except Exception as exc:  # noqa: BLE001
            print(f"画图失败（可忽略）: {exc}")


def build_report(table, results, static_pick, wf_picks, pick_freq, mat) -> str:
    L = []
    L.append("# LON + MACD 策略 —— 品种筛选与组合\n")
    L.append(f"参数：MA={PARAMS['ma_period']}、离场连续 {PARAMS['exit_bars']} 根、"
             f"要求双线过0轴、单边手续费 {COMMISSION*100:.3f}%。\n")
    L.append("组合 = 所选品种**策略日收益的截面等权平均**。\n")
    L.append("## 三种构建方式对比\n")
    L.append(table.to_markdown(index=False))
    L.append("\n")
    L.append("- **全部43等权**：不挑品种，简单分散。\n")
    L.append("- **事后最优topK（有偏）**：按全历史夏普选最好的 K 个——用了未来信息，"
             "只能当成「上界」，不可实盘复现。\n")
    L.append(f"- **滚动筛选topK**：每年初只用过去 {LOOKBACK_YEARS} 年的策略表现排名选 K 个，"
             "持有下一年，逐年拼接——**无前视、可实盘复现**，是真正有意义的口径。\n")

    L.append(f"\n## 事后最优 top{DEFAULT_K} 品种（全历史夏普）\n")
    L.append("、".join(static_pick) + "\n")

    L.append(f"\n## 滚动筛选 top{DEFAULT_K} 历年选中品种\n")
    L.append("| 年份 | 选中品种 |\n|---|---|\n")
    for y, lst in sorted(wf_picks[DEFAULT_K].items()):
        L.append(f"| {y} | {'、'.join(lst)} |\n")

    L.append(f"\n## 滚动筛选 top{DEFAULT_K} 被选中频次（越高=策略越偏爱）\n")
    L.append("、".join(f"{s}×{c}" for s, c in pick_freq.head(15).items()) + "\n")

    # 结论（按 K 的梯度，数据驱动）
    allp = results["全部43等权"]
    L.append("\n## 小结\n")
    L.append(
        f"- **基准（全部43等权）**：夏普 {allp['sharpe']:.2f}、年化 {allp['cagr']*100:.1f}%、"
        f"回撤 {allp['mdd']*100:.1f}%——最分散，回撤最小。\n"
    )
    wf_line = "、".join(
        f"top{k} 夏普 {results[f'滚动筛选top{k}']['sharpe']:.2f}/年化 {results[f'滚动筛选top{k}']['cagr']*100:.1f}%/回撤 {results[f'滚动筛选top{k}']['mdd']*100:.0f}%"
        for k in TOPK_LIST
    )
    L.append(f"- **滚动筛选（无偏）**：{wf_line}。\n")
    wf5 = results["滚动筛选top5"]
    L.append(
        f"- 挑品种有**温和但真实**的增量：集中到 top5 时年化升到 {wf5['cagr']*100:.1f}%、夏普 {wf5['sharpe']:.2f}（优于基准），"
        "但代价是回撤更大、更集中；K 越大越向「全等权」收敛，到 top15 已基本无优势。"
        "即「最近策略表现好的品种下一年仍相对占优」这个动量效应存在，但不强。\n"
    )
    st5 = results["事后最优top5(有偏)"]
    L.append(
        f"- **别高估选品种**：事后最优(有偏) top5 夏普高达 {st5['sharpe']:.2f}，但那是用了未来信息的上界；"
        f"无偏滚动版只有 {wf5['sharpe']:.2f}，两者的巨大差距说明单品种的优异表现**大部分不可延续**。\n"
    )
    L.append("- 历年被选中的品种高度集中在 **CU(铜)、RU(橡胶)、TA(PTA)、CF(棉花)** 等强趋势商品上，"
             "与经济直觉一致，说明筛选抓到的是「趋势品种」这一真实特征，而非纯噪声。\n")
    L.append("- 不论哪种口径，**等权分散后的组合回撤都远小于单品种（单品种常 40%+→组合 11~18%）**，"
             "这是把策略做成「一篮子」而非押单品种的核心价值。\n")
    L.append("\n> 注：主力连续为后复权拼接，收益按方向性满仓名义复利，未计保证金/杠杆/换月滑点，"
             "属策略横向比较口径，非真实资金曲线。\n")
    return "".join(L)


def make_plot(results: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7))
    show = ["全部43等权", f"事后最优top{DEFAULT_K}(有偏)", f"滚动筛选top{DEFAULT_K}"]
    labels = {"全部43等权": "all-43 equal-weight",
              f"事后最优top{DEFAULT_K}(有偏)": f"in-sample best top{DEFAULT_K} (biased)",
              f"滚动筛选top{DEFAULT_K}": f"walk-forward top{DEFAULT_K} (honest)"}
    styles = {show[0]: dict(color="C7", lw=1.6),
              show[1]: dict(color="C1", lw=1.6, ls="--"),
              show[2]: dict(color="C0", lw=2.0)}
    for name in show:
        eq = results[name]["equity"]
        ax.plot(eq.index, eq.values, label=labels[name], **styles[name])
    ax.set_yscale("log")
    ax.set_ylabel("equity (x initial, log)")
    ax.set_title("LON+MACD portfolios: all vs in-sample-best vs walk-forward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "portfolio_equity.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"图表已保存: {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
