"""
期货多空向量化回测引擎

与 `engine.BacktestEngine`（仅支持做多的事件驱动股票引擎）不同，本引擎面向
单标的、可双向持仓的期货连续合约回测：

- 输入价格序列和「每根 K 线收盘时确定的目标仓位状态」`state`（+1 多 / -1 空 / 0 空仓）；
- 信号在收盘确认，**下一根 K 线**才生效（position = state.shift(1)），避免未来函数；
- 以满仓名义本金、不加杠杆的方式按收盘价对收盘价计算策略收益，
  得到可与买入持有横向比较的净值曲线；
- 每次换仓按 `commission`（单边，名义额比例）扣费，反手按两次单边计。

注意：主力连续序列为后复权拼接，已尽量消除换月跳空；收益以百分比复利计，
不模拟保证金 / 强平，属于趋势策略的方向性收益评估。
"""

from typing import Dict, List

import numpy as np
import pandas as pd


def run_backtest(
    data: pd.DataFrame,
    commission: float = 0.0003,
    annualization: int = 252,
) -> Dict:
    """
    运行多空向量化回测。

    参数:
        data: 含 date/close/state 列的 DataFrame（state 为收盘确认的目标仓位）
        commission: 单边手续费率（名义额比例），换仓扣费
        annualization: 年化因子（日线 252）

    返回:
        含绩效指标、净值曲线和逐笔交易的字典。
    """
    df = data.reset_index(drop=True).copy()
    close = df["close"].astype(float)

    # 下一根 K 线才持有 -> position 较 state 滞后一根
    position = df["state"].shift(1).fillna(0).astype(float)

    pct = close.pct_change().fillna(0.0)
    gross_ret = position * pct

    # 换仓成本：仓位变化量 * 单边费率（多->空记为 |1-(-1)|=2 倍单边）
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * commission
    net_ret = gross_ret - cost

    equity = (1.0 + net_ret).cumprod()
    bh_equity = (1.0 + pct).cumprod()  # 买入持有基准

    df["position"] = position
    df["ret"] = net_ret
    df["equity"] = equity
    df["bh_equity"] = bh_equity

    total_return = equity.iloc[-1] - 1.0
    bh_return = bh_equity.iloc[-1] - 1.0

    n = len(df)
    years = n / annualization if n else 0.0
    cagr = equity.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 and equity.iloc[-1] > 0 else 0.0

    std = net_ret.std()
    sharpe = net_ret.mean() / std * np.sqrt(annualization) if std and std != 0 else 0.0

    downside = net_ret[net_ret < 0].std()
    sortino = net_ret.mean() / downside * np.sqrt(annualization) if downside and downside != 0 else 0.0

    cummax = equity.cummax()
    drawdown = equity / cummax - 1.0
    max_drawdown = drawdown.min()
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else 0.0

    trades = _extract_trades(df)
    if trades:
        tdf = pd.DataFrame(trades)
        wins = tdf[tdf["pnl_pct"] > 0]
        win_rate = len(wins) / len(tdf)
        avg_win = wins["pnl_pct"].mean() if len(wins) else 0.0
        losses = tdf[tdf["pnl_pct"] <= 0]
        avg_loss = losses["pnl_pct"].mean() if len(losses) else 0.0
        gross_win = wins["pnl_pct"].sum()
        gross_loss = abs(losses["pnl_pct"].sum())
        profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")
        avg_hold = tdf["bars"].mean()
    else:
        tdf = pd.DataFrame(columns=["direction", "entry_date", "exit_date", "pnl_pct", "bars"])
        win_rate = avg_win = avg_loss = profit_factor = avg_hold = 0.0

    exposure = (position != 0).mean()

    return {
        "total_return": total_return,
        "bh_return": bh_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "n_trades": len(tdf),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "avg_hold_bars": avg_hold,
        "exposure": exposure,
        "n_bars": n,
        "long_trades": int((tdf["direction"] == "long").sum()) if len(tdf) else 0,
        "short_trades": int((tdf["direction"] == "short").sum()) if len(tdf) else 0,
        "equity_curve": df[["date", "close", "position", "equity", "bh_equity"]],
        "trades": tdf,
    }


def _extract_trades(df: pd.DataFrame) -> List[Dict]:
    """根据 position 序列切分出逐笔交易（含多空），按名义复利计算每笔收益率。"""
    trades: List[Dict] = []
    position = df["position"].to_numpy()
    close = df["close"].to_numpy()
    dates = df["date"].to_numpy()
    n = len(df)

    i = 0
    while i < n:
        pos = position[i]
        if pos == 0:
            i += 1
            continue
        start = i
        while i < n and position[i] == pos:
            i += 1
        end = i - 1  # 最后一根持有该仓位的 K 线
        entry_price = close[start - 1] if start > 0 else close[start]
        exit_price = close[end]
        if pos > 0:
            pnl = exit_price / entry_price - 1.0
            direction = "long"
        else:
            pnl = entry_price / exit_price - 1.0
            direction = "short"
        trades.append(
            {
                "direction": direction,
                "entry_date": dates[start - 1] if start > 0 else dates[start],
                "exit_date": dates[end],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_pct": pnl,
                "bars": end - start + 1,
            }
        )
    return trades
