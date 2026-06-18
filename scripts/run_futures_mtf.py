"""
期货多周期 "看大做小" 回测 —— 主跑批
=====================================

围绕一个核心问题：**"看大周期做小周期" 能不能提高年化 / 风险调整后收益？**

跑批内容：
  1. 加载 5 大板块主力连续日线（本地缓存，2005~ 至今，最长 21 年）。
  2. 对每个品种生成 "看大做小" 多周期信号（周线 EMA 定方向 + 日线均线择时，
     方向一致才开多空），用目标波动率定仓，连续合约换月跳空做截断。
  3. 对照组：仅日线（ltf_only）、仅周线趋势（htf_only）、买入持有。
  4. **入场速度扫描**：在组合层面比较不同日线快慢下 mtf vs ltf —— 揭示
     "看大做小" 的价值主要在 *快入场* 上（过滤假突破、压回撤）。
  5. 分品种 / 分板块 / 全市场组合（按 15% 目标波动加杠杆，年化可比可交易）。
  6. 输出 results/ 下的 CSV、equity_curves.png、REPORT.md。

用法：
  python scripts/run_futures_mtf.py
  python scripts/run_futures_mtf.py --start 2015-01-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.futures import FUTURES_UNIVERSE, FuturesDataSource
from src.backtest.futures_engine import (
    FuturesBacktester,
    combine_portfolio,
    vol_target_weight,
)
from src.strategies.mtf import mtf_signal

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

# 主参数：日线 5/20 均线择时是收益/风险平衡较好的 "小周期"，周线 EMA 8/20 定 "大周期"。
PRIMARY = dict(ltf_method="ma_cross", ma_fast=5, ma_slow=20, ema_fast=8, ema_slow=20, rule="W")
TARGET_VOL = 0.15       # 单品种 & 组合目标年化波动 15%
MAX_LEV = 3.0           # 最大杠杆 3x（商品保证金约 10-15%）
COST = 0.0005           # 单边成本 万分之五（手续费 + 滑点）
RET_CLIP = 0.15         # 连续合约换月跳空截断阈值

# 板块拼音（matplotlib 默认无中文字体）
CAT_EN = {"黑色": "Ferrous", "有色": "BaseMetal", "贵金属": "Precious",
          "农产品": "Agri", "能化": "Energy/Chem"}


def _bt(daily: pd.DataFrame, mode: str, **override):
    params = {**PRIMARY, **override}
    prices = daily.set_index("date")["close"]
    ret = prices.pct_change().clip(-RET_CLIP, RET_CLIP).fillna(0.0)
    if mode == "buy_hold":
        weight = pd.Series(1.0, index=prices.index)
    else:
        sig = mtf_signal(daily, mode=mode, **params)
        weight = vol_target_weight(sig, ret, TARGET_VOL, 20, MAX_LEV)
    return FuturesBacktester(cost=COST, freq="D", ret_clip=RET_CLIP).run(prices, weight)


def load_universe(src, start, end):
    out = {}
    for cat, syms in FUTURES_UNIVERSE.items():
        for symbol, name in syms:
            daily = src.get_daily(symbol)
            if daily is None or len(daily) < 300:
                continue
            if start:
                daily = daily[daily["date"] >= pd.Timestamp(start)]
            if end:
                daily = daily[daily["date"] <= pd.Timestamp(end)]
            daily = daily.reset_index(drop=True)
            if len(daily) >= 300:
                out[symbol] = (cat, name, daily)
    return out


def entry_speed_sweep(universe):
    """组合层面：不同日线快慢，mtf vs ltf_only 的对比（看大做小价值所在）。"""
    rows = []
    for fast, slow, tag in [(3, 10, "快 ma3/10"), (5, 20, "中 ma5/20"),
                            (10, 30, "慢 ma10/30"), (20, 60, "更慢 ma20/60")]:
        for mode in ["ltf_only", "mtf"]:
            res = {s: _bt(d, mode, ma_fast=fast, ma_slow=slow)
                   for s, (_, _, d) in universe.items()}
            p = combine_portfolio(res, target_vol=TARGET_VOL, max_leverage=MAX_LEV)
            rows.append({"入场": tag, "模式": mode, "年化": p["cagr"],
                         "夏普": p["sharpe"], "回撤": p["max_drawdown"],
                         "Calmar": p["calmar"]})
    return pd.DataFrame(rows)


def run(start, end, only_cat):
    src = FuturesDataSource()
    universe = load_universe(src, start, end)
    if only_cat:
        universe = {s: v for s, v in universe.items() if v[0] == only_cat}

    rows, mtf_res, ltf_res, cat_res = [], {}, {}, {}
    for symbol, (cat, name, daily) in universe.items():
        r_mtf = _bt(daily, "mtf")
        r_ltf = _bt(daily, "ltf_only")
        r_bh = _bt(daily, "buy_hold")
        mtf_res[symbol] = r_mtf
        ltf_res[symbol] = r_ltf
        cat_res.setdefault(cat, {})[symbol] = r_mtf
        rows.append({
            "板块": cat, "品种": symbol, "名称": name, "年数": round(r_mtf["years"], 1),
            "MTF年化": r_mtf["cagr"], "MTF夏普": r_mtf["sharpe"],
            "MTF回撤": r_mtf["max_drawdown"], "MTF_Calmar": r_mtf["calmar"],
            "小周期年化": r_ltf["cagr"], "小周期夏普": r_ltf["sharpe"],
            "买持年化": r_bh["cagr"], "胜率": r_mtf["win_rate"], "年换手": r_mtf["turnover"],
        })
    by_symbol = pd.DataFrame(rows).sort_values("MTF夏普", ascending=False)

    # 分板块组合
    cat_rows, cat_ports = [], {}
    for cat, res in cat_res.items():
        port = combine_portfolio(res, target_vol=TARGET_VOL, max_leverage=MAX_LEV)
        cat_ports[cat] = port
        sub = by_symbol[by_symbol["板块"] == cat]
        cat_rows.append({
            "板块": cat, "品种数": len(res),
            "组合年化": port["cagr"], "组合夏普": port["sharpe"],
            "组合回撤": port["max_drawdown"], "组合Calmar": port["calmar"],
            "MTF均值夏普": sub["MTF夏普"].mean(), "小周期均值夏普": sub["小周期夏普"].mean(),
        })
    by_category = pd.DataFrame(cat_rows).sort_values("组合夏普", ascending=False)

    all_mtf = combine_portfolio(mtf_res, target_vol=TARGET_VOL, max_leverage=MAX_LEV)
    all_ltf = combine_portfolio(ltf_res, target_vol=TARGET_VOL, max_leverage=MAX_LEV)
    sweep = entry_speed_sweep(universe)

    _report(by_symbol, by_category, all_mtf, all_ltf, cat_ports, sweep, start, end)
    return by_symbol, by_category, all_mtf, cat_ports, sweep


def _pct(x):
    return f"{x*100:.1f}%"


def _report(by_symbol, by_category, all_mtf, all_ltf, cat_ports, sweep, start, end):
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 220)

    print("\n" + "=" * 72)
    print("入场速度扫描（全市场组合）：看大做小的价值集中在『快入场』")
    print("=" * 72)
    sw = sweep.copy()
    for c in ["年化", "回撤"]:
        sw[c] = sw[c].map(_pct)
    for c in ["夏普", "Calmar"]:
        sw[c] = sw[c].map(lambda v: f"{v:.2f}")
    print(sw.to_string(index=False))

    print("\n" + "=" * 72)
    print("分板块组合（按 15% 目标波动加杠杆的等权 CTA 组合）")
    print("=" * 72)
    cd = by_category.copy()
    for c in ["组合年化", "组合回撤"]:
        cd[c] = cd[c].map(_pct)
    for c in ["组合夏普", "组合Calmar", "MTF均值夏普", "小周期均值夏普"]:
        cd[c] = cd[c].map(lambda v: f"{v:.2f}")
    print(cd.to_string(index=False))

    print("\n" + "=" * 72)
    print("全市场组合：看大做小(MTF) vs 仅日线(LTF)")
    print("=" * 72)
    for tag, p in [("MTF 看大做小", all_mtf), ("LTF 仅日线", all_ltf)]:
        s = p.stats
        print(f"  {tag:14}  年化 {_pct(s['cagr']):>7} | 夏普 {s['sharpe']:.2f} | "
              f"回撤 {_pct(s['max_drawdown']):>7} | Calmar {s['calmar']:.2f} | "
              f"波动 {_pct(s['ann_vol'])} | {s['years']:.1f} 年")

    print("\n" + "=" * 72)
    print("单品种明细（按 MTF 夏普排序，前 15）")
    print("=" * 72)
    disp = by_symbol.copy()
    for c in ["MTF年化", "MTF回撤", "小周期年化", "买持年化", "胜率"]:
        disp[c] = disp[c].map(_pct)
    for c in ["MTF夏普", "MTF_Calmar", "小周期夏普", "年换手"]:
        disp[c] = disp[c].map(lambda v: f"{v:.2f}")
    print(disp.head(15).to_string(index=False))

    by_symbol.to_csv(RESULTS / "summary_by_symbol.csv", index=False)
    by_category.to_csv(RESULTS / "summary_by_category.csv", index=False)
    sweep.to_csv(RESULTS / "entry_speed_sweep.csv", index=False)
    _plot(all_mtf, all_ltf, cat_ports)
    _write_md(by_symbol, by_category, all_mtf, all_ltf, sweep, start, end)
    print(f"\n✅ 结果已保存到 {RESULTS}/（CSV / equity_curves.png / REPORT.md）")


def _plot(all_mtf, all_ltf, cat_ports):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(11, 6))
        for cat, port in cat_ports.items():
            ax.plot(port.equity.index, port.equity.values,
                    label=CAT_EN.get(cat, cat), alpha=0.75)
        ax.plot(all_mtf.equity.index, all_mtf.equity.values,
                label="ALL (MTF)", color="black", lw=2.5)
        ax.plot(all_ltf.equity.index, all_ltf.equity.values,
                label="ALL (LTF only)", color="gray", lw=1.5, ls="--")
        ax.set_yscale("log")
        ax.set_title("Multi-Timeframe Top-Down Futures CTA (vol-targeted 15%)")
        ax.set_ylabel("Equity (log, start=1.0)")
        ax.legend(ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS / "equity_curves.png", dpi=110)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠️ 绘图跳过: {exc!r}")


def _write_md(by_symbol, by_category, all_mtf, all_ltf, sweep, start, end):
    L = ["# 期货多周期 “看大做小” 回测报告\n"]
    L.append(f"- 区间：{start or '全历史'} ~ {end or '至今'}；"
             f"单品种&组合目标波动 {int(TARGET_VOL*100)}%，最大杠杆 {MAX_LEV}x，"
             f"单边成本 {COST*1e4:.0f}bp，换月跳空截断 ±{int(RET_CLIP*100)}%")
    L.append("- 信号：周线 EMA(8/20) 定方向；日线 MA(5/20) 择时；方向一致才开仓（多空双向）\n")

    s = all_mtf.stats
    L.append("## 全市场等权组合（看大做小）\n")
    L.append(f"- **年化 {_pct(s['cagr'])}，夏普 {s['sharpe']:.2f}，最大回撤 "
             f"{_pct(s['max_drawdown'])}，Calmar {s['calmar']:.2f}**，"
             f"波动 {_pct(s['ann_vol'])}（{s['years']:.1f} 年）")
    sl = all_ltf.stats
    L.append(f"- 对照（仅日线趋势）：年化 {_pct(sl['cagr'])}，夏普 {sl['sharpe']:.2f}，"
             f"回撤 {_pct(sl['max_drawdown'])}，Calmar {sl['calmar']:.2f}\n")

    L.append("## 入场速度扫描：看大做小的价值在『快入场』\n")
    L.append("| 日线入场 | 模式 | 年化 | 夏普 | 回撤 | Calmar |")
    L.append("|---|---|---|---|---|---|")
    for _, r in sweep.iterrows():
        L.append(f"| {r['入场']} | {r['模式']} | {_pct(r['年化'])} | {r['夏普']:.2f} | "
                 f"{_pct(r['回撤'])} | {r['Calmar']:.2f} |")
    L.append("\n> 结论：日线入场越快，单用日线越容易被假信号打损（夏普低）；叠加大周期"
             "趋势过滤后，快入场的夏普/回撤明显改善。入场越慢，大周期过滤的增量越小。\n")

    L.append("## 分板块组合\n")
    L.append("| 板块 | 品种数 | 组合年化 | 组合夏普 | 组合回撤 | Calmar | MTF均值夏普 | 仅日线均值夏普 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for _, r in by_category.iterrows():
        L.append(f"| {r['板块']} | {int(r['品种数'])} | {_pct(r['组合年化'])} | "
                 f"{r['组合夏普']:.2f} | {_pct(r['组合回撤'])} | {r['组合Calmar']:.2f} | "
                 f"{r['MTF均值夏普']:.2f} | {r['小周期均值夏普']:.2f} |")
    L.append("\n> 板块底层逻辑差异：有色/能化/黑色（工业品，受经济周期与产业链驱动）和"
             "贵金属（宏观/货币属性）趋势性强、顺势策略有效；农产品受天气、收储等供给端"
             "随机冲击，趋势性弱、易反复，顺势策略最弱。\n")

    L.append("## 单品种前 15（按 MTF 夏普）\n")
    L.append("| 板块 | 品种 | 名称 | MTF年化 | MTF夏普 | 回撤 | Calmar | 仅日线夏普 | 买入持有年化 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in by_symbol.head(15).iterrows():
        L.append(f"| {r['板块']} | {r['品种']} | {r['名称']} | {_pct(r['MTF年化'])} | "
                 f"{r['MTF夏普']:.2f} | {_pct(r['MTF回撤'])} | {r['MTF_Calmar']:.2f} | "
                 f"{r['小周期夏普']:.2f} | {_pct(r['买持年化'])} |")

    L.append("\n## 一句话总结\n")
    L.append("“看大周期做小周期” 不是万能的收益放大器，而是 **风险/择时过滤器**："
             "它在 *快入场* 和 *强趋势板块（工业品/贵金属）* 上能提升夏普、压低回撤；"
             "在慢入场或弱趋势板块（农产品）上增量有限甚至拖累收益。要做到更高且稳健的"
             "年化，真正的杠杆来自 **多品种分散 + 波动率目标定仓 + 板块筛选**，"
             "再用 “看大做小” 给快择时上一道保险。")
    (RESULTS / "REPORT.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--category", default=None)
    args = ap.parse_args()
    run(args.start, args.end, args.category)
