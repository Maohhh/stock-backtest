"""求魔策略诊断: 分离「入场是否有edge」与「出场是否毁掉赢家」。

A) 入场前向收益: 信号后 h 根的毛收益(不受出场逻辑影响), 与同向"全bar基准"比;
   多信号收益 > 基准 才说明入场有预测力, 空信号(方向取负)同理。
B) 实现交易赢亏不对称 + 右尾依赖 + 按出场原因盈亏归因。
"""
import glob
import numpy as np
import pandas as pd
import backtest_qiumo as bt

WARMUP = 1500
H = [10, 20, 40, 80, 160]
files = sorted(glob.glob("data_futures/15min/*.parquet"))

LF = {h: [] for h in H}; SF = {h: [] for h in H}; AF = {h: [] for h in H}
for f in files:
    df = pd.read_parquet(f).sort_values("datetime").reset_index(drop=True)
    if len(df) < WARMUP + 200:
        continue
    ind = bt.compute_indicators(df, use_macro=True)
    c = ind["close"].to_numpy(); n = len(c); warm = np.arange(n) >= WARMUP
    bb = ind["buy_base"].to_numpy() & warm; sb = ind["sell_base"].to_numpy() & warm
    for h in H:
        fwd = np.full(n, np.nan); fwd[:n - h] = c[h:] / c[:n - h] - 1
        m = ~np.isnan(fwd)
        LF[h].append(fwd[bb & m]); SF[h].append(-fwd[sb & m]); AF[h].append(fwd[warm & m])

print("=" * 74)
print("A) 入场信号前向收益(毛) vs 同向全bar基准  —— 超额>0 才说明入场有edge")
print("=" * 74)
print(f"{'h':>4} | {'多·均值':>9} {'多·胜率':>7} {'基准':>8} {'超额':>9} | {'空·均值':>9} {'空·胜率':>7}")
for h in H:
    l = np.concatenate(LF[h]); s = np.concatenate(SF[h]); a = np.concatenate(AF[h])
    print(f"{h:>4} | {l.mean():>9.4%} {(l > 0).mean():>7.1%} {a.mean():>8.4%} "
          f"{l.mean() - a.mean():>9.4%} | {s.mean():>9.4%} {(s > 0).mean():>7.1%}")

t = pd.read_csv("data_futures/backtest_qiumo/trades.csv")
full = t.groupby(["product", "entry_i"]).agg(ret=("ret", "sum"), hold=("hold_bars", "max")).reset_index()
w = full[full.ret > 0].ret; l = full[full.ret <= 0].ret
print("=" * 74)
print("B) 实现交易(整仓)赢亏不对称 + 右尾依赖")
print("=" * 74)
print(f"笔数={len(full)}  胜率={(full.ret > 0).mean():.1%}  平均盈利={w.mean():.2%}  "
      f"平均亏损={l.mean():.2%}  盈亏比={w.mean() / -l.mean():.2f}")
print(f"赢家中位持仓={full[full.ret > 0].hold.median():.0f}根  输家中位持仓={full[full.ret <= 0].hold.median():.0f}根")
top = int(len(full) * 0.05); fs = full.sort_values("ret", ascending=False)
print(f"最赚5%({top}笔)合计={fs.head(top).ret.sum():.2f}  其余95%合计={fs.iloc[top:].ret.sum():.2f}")
print("\n按出场原因盈亏归因:")
g = t.groupby("reason").agg(笔数=("ret_unit", "size"), 均收益=("ret_unit", "mean"),
                            合计贡献=("ret", "sum")).sort_values("合计贡献")
print(g.round(4).to_string())
