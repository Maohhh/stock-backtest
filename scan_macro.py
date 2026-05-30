"""扫描宏观趋势门均线长度(MACRO)对求魔策略的影响。

为公平对比, 所有变体在同一批 K 线上交易(统一 1500 根预热), 仅改宏观均线长度。
"""
import glob
import os

import numpy as np
import pandas as pd

import backtest_qiumo as bt

WARMUP = 1500          # 统一预热, 保证各变体可交易 K 线一致
MACROS = [0, 600, 900, 1200, 1500]   # 0 = 关闭宏观门

files = sorted(glob.glob("data_futures/15min/*.parquet"))

rows = []
per_product = {}       # macro -> {product: total_ret}
for macro in MACROS:
    use_macro = macro > 0
    bt.MACRO = macro if use_macro else WARMUP
    all_trades = []
    prod_ret = {}
    for f in files:
        product = os.path.basename(f)[:-8]
        df = pd.read_parquet(f).sort_values("datetime").reset_index(drop=True)
        if len(df) < WARMUP + 50:
            continue
        ind = bt.compute_indicators(df, use_macro=use_macro)
        ind["warm"] = np.arange(len(ind)) >= WARMUP      # 统一预热口径
        trades = bt.simulate(ind, product)
        all_trades.extend(trades)
        s = bt.summarize(trades)
        prod_ret[product] = s.get("total_ret", 0.0) if s else 0.0
    per_product[macro] = prod_ret
    agg = bt.summarize(all_trades)
    label = "关闭" if macro == 0 else str(macro)
    rows.append({
        "MACRO": label,
        "品种数": sum(1 for v in prod_ret.values() if v != 0),
        "交易数": agg["trades"],
        "胜率": agg["win_rate"],
        "PF": agg["pf"],
        "等权合计": agg["total_ret"],
        "单笔均收益": agg["avg_ret"],
        "盈利品种": sum(1 for v in prod_ret.values() if v > 0),
    })

res = pd.DataFrame(rows)
print("=" * 78)
print(f"宏观门均线长度扫描  (统一{WARMUP}根预热, 同一批K线对比)")
print("=" * 78)
with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
    print(res.to_string(index=False))
print("=" * 78)

os.makedirs("data_futures/backtest_qiumo", exist_ok=True)
res.to_csv("data_futures/backtest_qiumo/macro_scan.csv", index=False)
print("已写入: data_futures/backtest_qiumo/macro_scan.csv")
