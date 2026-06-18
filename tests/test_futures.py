"""期货多周期子系统单元测试（不依赖网络，用合成数据）。"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.futures_engine import (
    FuturesBacktester,
    combine_portfolio,
    vol_target_weight,
)
from src.data.futures import resample_ohlc, symbol_category
from src.strategies.mtf import htf_bias, ltf_signal, mtf_signal


def _trend_df(n=600, slope=0.001, seed=0):
    """构造一条带上升趋势的日线数据。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    ret = slope + rng.normal(0, 0.01, n)
    close = 1000 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    return pd.DataFrame({"date": dates, "open": close, "high": high,
                         "low": low, "close": close, "volume": 1000})


def test_resample_weekly_aggregates():
    df = _trend_df(60)
    wk = resample_ohlc(df, "W")
    assert len(wk) < len(df)
    assert {"date", "open", "high", "low", "close"}.issubset(wk.columns)
    # 周线高点不低于其内日线收盘
    assert (wk["high"] >= wk["close"]).all()


def test_long_only_uptrend_profits():
    df = _trend_df(slope=0.0015)
    prices = df.set_index("date")["close"]
    weight = pd.Series(1.0, index=prices.index)
    res = FuturesBacktester().run(prices, weight)
    assert res["cagr"] > 0
    assert res.equity.iloc[-1] > 1.0


def test_vol_target_caps_leverage():
    df = _trend_df()
    ret = df.set_index("date")["close"].pct_change().fillna(0)
    sig = pd.Series(1.0, index=ret.index)
    w = vol_target_weight(sig, ret, target_vol=0.15, max_leverage=2.0)
    assert w.abs().max() <= 2.0 + 1e-9
    assert (w >= 0).all()  # 多头信号不应产生负仓位


def test_mtf_filters_against_htf():
    """mtf 信号必须是 ltf 与 htf 方向的交集（绝不逆大周期开仓）。"""
    df = _trend_df()
    bias = htf_bias(df)
    ltf = ltf_signal(df, method="ma_cross")
    sig = mtf_signal(df, ltf_method="ma_cross", mode="mtf")
    # 任何非零仓位都必须与大周期同向
    nz = sig[sig != 0]
    assert (np.sign(nz.values) == np.sign(bias.reindex(nz.index).values)).all()


def test_rollover_clip():
    """单日极端跳空应被截断，不至于把净值打到 0。"""
    df = _trend_df(200)
    prices = df.set_index("date")["close"].copy()
    prices.iloc[100] *= 2.0  # 制造 +100% 假跳空
    weight = pd.Series(-1.0, index=prices.index)  # 满仓做空吃这个跳空
    res = FuturesBacktester(ret_clip=0.15).run(prices, weight)
    assert res.equity.iloc[-1] > 0  # 截断后不爆仓


def test_combine_portfolio_vol_target():
    a = FuturesBacktester().run(_trend_df(seed=1).set_index("date")["close"],
                                pd.Series(1.0, index=_trend_df(seed=1)["date"]))
    b = FuturesBacktester().run(_trend_df(seed=2).set_index("date")["close"],
                                pd.Series(1.0, index=_trend_df(seed=2)["date"]))
    port = combine_portfolio({"a": a, "b": b}, target_vol=0.15)
    # 组合年化波动应接近目标（容许区间）
    assert 0.05 < port["ann_vol"] < 0.30


def test_symbol_category_lookup():
    assert symbol_category("RB0") == "黑色"
    assert symbol_category("CU0") == "有色"
    assert symbol_category("M0") == "农产品"
    assert symbol_category("NOPE") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
