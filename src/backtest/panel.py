"""
横截面多因子面板回测引擎
==========================

把视角从 "单品种方向" 升级到 "一篮子品种的横截面"。这是把年化做高且稳健的
关键 —— 兄弟分支已穷尽证明：**单标的方向性信号扣成本后不稳健**，真正能赚钱
的是 **横截面多空（买强空弱 / 买高 carry 空低 carry）**，赚的是品种间价差收敛
和板块内中性对冲的钱，夏普显著更高，再用杠杆把年化放大。

本引擎处理一个 "面板"（dates × instruments）：
- 输入每个品种的日收益矩阵和因子打分矩阵；
- 每个品种按其自身波动反比定权（风险平价），可选板块内中性化（多空对冲）；
- 组合按 **目标波动率** 加杠杆，使年化可比、杠杆显性；
- 输出净值、夏普、回撤，以及 **年化-杠杆前沿**。

因子：
- TS 动量（时序）：每个品种看自己过去 N 日涨跌定方向。
- XS 动量（横截面）：每天把所有品种按过去 N 日收益排序，买强空弱。
- Carry（展期结构）：买高 carry（backwardation）空低 carry（contango）。
- 组合：多因子打分等权合成。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# 43 个主力连续品种的板块归属（含金融期货），用于板块内中性化。
SECTOR = {
    "黑色": ["RB0", "HC0", "I0", "J0", "JM0", "SM0", "SF0", "SS0"],
    "有色": ["CU0", "AL0", "ZN0", "NI0", "SN0", "PB0", "BC0", "AO0"],
    "贵金属": ["AU0", "AG0"],
    "农产品": ["M0", "Y0", "C0", "A0", "P0", "CF0", "SR0", "OI0", "RM0",
             "CS0", "JD0", "AP0", "B0"],
    "能化": ["TA0", "MA0", "PP0", "L0", "V0", "RU0", "BU0", "FG0", "EG0",
            "SA0", "FU0", "SC0", "PG0", "EB0"],
    "金融": ["IF0", "IC0", "IH0", "IM0", "T0", "TF0", "TL0"],
}
SYM2SECTOR = {s: k for k, syms in SECTOR.items() for s in syms}


# --------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------- #
def load_daily_panel(datadir: str = "data_futures/sina_daily_main",
                     symbols: Optional[List[str]] = None,
                     start: Optional[str] = None) -> pd.DataFrame:
    """加载日线收盘价面板，列=品种，行=日期。"""
    d = Path(datadir)
    files = sorted(d.glob("*.csv"))
    out = {}
    for f in files:
        sym = f.stem
        if symbols and sym not in symbols:
            continue
        df = pd.read_csv(f)
        # 兼容两种列名（d/o/h/l/c/v 或 date/...）
        cols = {c.lower(): c for c in df.columns}
        dcol = cols.get("d") or cols.get("date")
        ccol = cols.get("c") or cols.get("close")
        if dcol is None or ccol is None:
            continue
        s = pd.Series(pd.to_numeric(df[ccol], errors="coerce").values,
                      index=pd.to_datetime(df[dcol]))
        s = s[~s.index.duplicated(keep="last")].sort_index()
        out[sym] = s
    panel = pd.DataFrame(out).sort_index()
    if start:
        panel = panel[panel.index >= pd.Timestamp(start)]
    return panel


def load_carry_panel(datadir: str = "data_futures/carry",
                     start: Optional[str] = None) -> pd.DataFrame:
    """加载 carry 面板（列=品种带 0 后缀，对齐日线命名）。"""
    d = Path(datadir)
    out = {}
    for f in sorted(d.glob("*.parquet")):
        sym = f.stem + "0"  # carry 用 'CU'，日线用 'CU0'
        df = pd.read_parquet(f)
        if "carry" not in df.columns or "date" not in df.columns:
            continue
        s = pd.Series(df["carry"].values, index=pd.to_datetime(df["date"]))
        s = s[~s.index.duplicated(keep="last")].sort_index()
        out[sym] = s
    panel = pd.DataFrame(out).sort_index()
    if start:
        panel = panel[panel.index >= pd.Timestamp(start)]
    return panel


# --------------------------------------------------------------------- #
# 因子打分（返回 dates × instruments 的打分矩阵）
# --------------------------------------------------------------------- #
def _zscore_xs(df: pd.DataFrame) -> pd.DataFrame:
    """逐行（每个交易日）做横截面 z-score。"""
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def ts_momentum_score(prices: pd.DataFrame, lookbacks=(60, 120, 250)) -> pd.DataFrame:
    """时序动量：每个品种过去 N 日收益的符号，多周期平均。"""
    score = sum(np.sign(prices / prices.shift(lb) - 1.0) for lb in lookbacks)
    return score / len(lookbacks)


def xs_momentum_score(prices: pd.DataFrame, lookbacks=(60, 120, 250),
                      skip: int = 0, sector_neutral: bool = True) -> pd.DataFrame:
    """横截面动量：过去 N 日收益的横截面 z-score（多周期平均）。

    skip>0 时跳过最近 skip 日（避开短期反转污染），即经典 "12-1 跳月动量"。
    实测国内商品 **必须跳月**，否则动量被短期反转吃掉（IC≈0 甚至为负）。
    """
    raw = sum((prices.shift(skip) / prices.shift(lb) - 1.0)
              for lb in lookbacks) / len(lookbacks)
    if sector_neutral:
        return _sector_demean(_zscore_xs(raw))
    return _zscore_xs(raw)


def xs_reversal_score(prices: pd.DataFrame, lookback: int = 5,
                      sector_neutral: bool = True) -> pd.DataFrame:
    """横截面短期反转：买近期跌得多的、空涨得多的（国内商品最强的横截面信号）。"""
    raw = -(prices / prices.shift(lookback) - 1.0)
    if sector_neutral:
        return _sector_demean(_zscore_xs(raw))
    return _zscore_xs(raw)


def carry_score(carry: pd.DataFrame, sector_neutral: bool = True) -> pd.DataFrame:
    """Carry 因子：carry 的横截面 z-score。"""
    z = _zscore_xs(carry)
    return _sector_demean(z) if sector_neutral else z


def _sector_demean(score: pd.DataFrame) -> pd.DataFrame:
    """板块内中性化：每个板块内部减去板块均值（板块中性多空）。"""
    out = score.copy()
    for sec, syms in SECTOR.items():
        cols = [s for s in syms if s in score.columns]
        if len(cols) >= 2:
            out[cols] = score[cols].sub(score[cols].mean(axis=1), axis=0)
    return out


def combine_scores(scores: Dict[str, pd.DataFrame],
                   weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """多因子等权（或加权）合成，合成后再做一次横截面标准化。"""
    weights = weights or {k: 1.0 for k in scores}
    common = None
    for df in scores.values():
        common = df.index if common is None else common.union(df.index)
    cols = None
    for df in scores.values():
        cols = df.columns if cols is None else cols.union(df.columns)
    agg = pd.DataFrame(0.0, index=common, columns=cols)
    wsum = pd.DataFrame(0.0, index=common, columns=cols)
    for k, df in scores.items():
        a = df.reindex(index=common, columns=cols)
        mask = a.notna()
        agg = agg.add((a.fillna(0.0) * weights[k]), fill_value=0.0)
        wsum = wsum.add(mask * weights[k], fill_value=0.0)
    return agg.div(wsum.replace(0, np.nan))


# --------------------------------------------------------------------- #
# 面板回测
# --------------------------------------------------------------------- #
def score_to_weights(score: pd.DataFrame, returns: pd.DataFrame,
                     vol_window: int = 40, gross: float = 2.0) -> pd.DataFrame:
    """把因子打分转成风险平价权重（按品种波动反比缩放，控制总杠杆 gross）。"""
    inst_vol = returns.rolling(vol_window).std().shift(1)
    inst_vol = inst_vol.replace(0, np.nan)
    raw = score / inst_vol                       # 风险平价：低波动品种给更大名义
    raw = raw.reindex_like(returns)
    # 每日归一化到固定总杠杆（gross = sum|w|）
    abs_sum = raw.abs().sum(axis=1).replace(0, np.nan)
    w = raw.div(abs_sum, axis=0) * gross
    return w.fillna(0.0)


def panel_backtest(returns: pd.DataFrame, weights: pd.DataFrame,
                   cost: float = 0.0005, target_vol: Optional[float] = None,
                   ann: float = 252.0, ret_clip: float = 0.15) -> dict:
    """面板回测：返回净值与绩效。可选组合层面目标波动率加杠杆。"""
    returns = returns.clip(-ret_clip, ret_clip).fillna(0.0)
    w = weights.reindex_like(returns).fillna(0.0)
    eff = w.shift(1).fillna(0.0)
    gross_ret = (eff * returns).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    port = gross_ret - turnover * cost

    if target_vol is not None:
        realized = port.rolling(40).std().shift(1) * np.sqrt(ann)
        lev = (target_vol / realized.replace(0, np.nan)).clip(upper=10.0).fillna(0.0)
        port = (lev * port).fillna(0.0)

    equity = (1.0 + port).cumprod()
    return _panel_metrics(port, equity, turnover, ann)


def _panel_metrics(port: pd.Series, equity: pd.Series,
                   turnover: pd.Series, ann: float) -> dict:
    n = len(port)
    years = n / ann if n else 0.0
    if equity.iloc[-1] <= 0 or years <= 0:
        return {"cagr": -1.0, "sharpe": 0.0, "ann_vol": 0.0,
                "max_drawdown": -1.0, "calmar": 0.0, "years": years,
                "turnover": 0.0, "equity": equity, "returns": port}
    cagr = equity.iloc[-1] ** (1 / years) - 1
    ann_vol = port.std() * np.sqrt(ann)
    sharpe = port.mean() / port.std() * np.sqrt(ann) if port.std() > 0 else 0.0
    dd = (equity / equity.cummax() - 1.0).min()
    calmar = cagr / abs(dd) if dd < 0 else 0.0
    return {"cagr": float(cagr), "sharpe": float(sharpe), "ann_vol": float(ann_vol),
            "max_drawdown": float(dd), "calmar": float(calmar), "years": float(years),
            "turnover": float(turnover.sum() / years),
            "equity": equity, "returns": port}


def leverage_frontier(port_ret: pd.Series, vols=(0.10, 0.15, 0.20, 0.30, 0.40),
                      ann: float = 252.0) -> pd.DataFrame:
    """把一条策略收益按不同目标波动率重新加杠杆，展示年化-回撤前沿。"""
    rows = []
    base_vol = port_ret.std() * np.sqrt(ann)
    for tv in vols:
        realized = port_ret.rolling(40).std().shift(1) * np.sqrt(ann)
        lev = (tv / realized.replace(0, np.nan)).clip(upper=10.0).fillna(0.0)
        r = (lev * port_ret).fillna(0.0)
        eq = (1.0 + r).cumprod()
        m = _panel_metrics(r, eq, pd.Series(0.0, index=r.index), ann)
        rows.append({"目标波动": tv, "约杠杆": tv / base_vol if base_vol else np.nan,
                     "年化": m["cagr"], "夏普": m["sharpe"],
                     "最大回撤": m["max_drawdown"], "Calmar": m["calmar"]})
    return pd.DataFrame(rows)
