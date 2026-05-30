"""品种筛选: 只做"匹配"的品种是否真能提升 —— 必须样本外验证。

陷阱: 按历史PnL挑品种再报该篮子业绩=选择偏差。诚实做法:
  仅用【训练期】挑品种, 固定该篮子后看【样本外】是否仍好。

检验三件事:
  1) 单品种 训练夏普 vs 样本外夏普 的相关性(>0 才说明"过去好"能预测"将来好")
  2) 按训练期表现挑 top-K 篮子, 看样本外组合夏普 vs 全品种
  3) 按"结构特征"(方差比VR<1=均值回归友好)挑篮子, 看样本外是否更好
配置: 最优配置 等额+5ATR止损+宏观。
"""
import numpy as np
import pandas as pd
import optimize_full as of

CFG = dict(sizing="equal", stop_k=5.0, confirm=False, macro=True)

P = of.load()
alld = sorted(set().union(*[set(d["dt"].dt.normalize()) for d in P]))
split_date = alld[int(len(alld) * 0.70)]


def sharpe(r):
    r = r.dropna()
    return r.mean() / r.std() * np.sqrt(252) if (len(r) > 5 and r.std() > 0) else 0.0


rows = []
daily_by_prod = {}
for d in P:
    rb, sz, tr = of.sim(d, **CFG)
    s = pd.Series(rb, index=d["dt"])
    dly = s.groupby(s.index.normalize()).sum()
    daily_by_prod[d["prod"]] = dly
    tr_d = dly[dly.index <= split_date]; te_d = dly[dly.index > split_date]
    # 结构特征: 训练期 bar 收益的方差比与一阶自相关
    c = d["c"]; mask = np.arange(len(c)) >= of.WARMUP
    ret1 = pd.Series(c).pct_change().to_numpy()
    # 训练段
    spl_i = int(np.searchsorted(d["dt"].values,
                                np.datetime64(split_date)))
    r_tr = ret1[of.WARMUP:spl_i]
    r_tr = r_tr[~np.isnan(r_tr)]
    if len(r_tr) > 100:
        q = 10
        v1 = np.var(r_tr)
        rq = np.add.reduceat(r_tr[:len(r_tr) // q * q],
                             np.arange(0, len(r_tr) // q * q, q))
        vr = (np.var(rq) / q) / v1 if v1 > 0 else np.nan
        ac1 = np.corrcoef(r_tr[:-1], r_tr[1:])[0, 1]
    else:
        vr = ac1 = np.nan
    rows.append({"prod": d["prod"], "tr_sharpe": sharpe(tr_d),
                 "te_sharpe": sharpe(te_d), "trades_tr": (tr_d != 0).sum(),
                 "vr": vr, "ac1": ac1})

R = pd.DataFrame(rows)


def basket_oos(prods, label):
    sub = pd.DataFrame({p: daily_by_prod[p] for p in prods}).fillna(0.0).mean(axis=1)
    te = sub[sub.index > split_date]
    eq = (1 + te).cumprod()
    print(f"{label:<34} 品种数={len(prods):>3}  OOS夏普={sharpe(te):>5.2f}  "
          f"OOS收益={eq.iloc[-1]-1:>+6.2%}")


print(f"品种={len(P)}  训练/样本外切分={split_date.date()}")
print("=" * 84)
print("1) 单品种 训练夏普 vs 样本外夏普 的相关性  (>0 才说明历史能选未来)")
print("=" * 84)
valid = R.dropna(subset=["tr_sharpe", "te_sharpe"])
corr = valid["tr_sharpe"].corr(valid["te_sharpe"])
rank_corr = valid["tr_sharpe"].rank().corr(valid["te_sharpe"].rank())
print(f"Pearson={corr:+.3f}   Spearman(秩)={rank_corr:+.3f}")
print(f"训练为正的品种里, 样本外仍为正的比例: "
      f"{(valid[valid.tr_sharpe>0].te_sharpe>0).mean():.0%} "
      f"({(valid.tr_sharpe>0).sum()}个训练为正)")

print("=" * 84)
print("2) 按【训练期】表现挑篮子, 看【样本外】组合夏普")
print("=" * 84)
basket_oos(list(R["prod"]), "全品种(基准)")
order = R.sort_values("tr_sharpe", ascending=False)["prod"].tolist()
for k in [10, 20, 30]:
    basket_oos(order[:k], f"训练夏普 Top{k}")
basket_oos(R[R.tr_sharpe > 0.3]["prod"].tolist(), "训练夏普>0.3")

print("=" * 84)
print("3) 按【结构特征·方差比VR<1=均值回归友好】挑篮子(训练期算), 看样本外")
print("=" * 84)
vr_ok = R[(R.vr < 0.9)].dropna(subset=["vr"])
basket_oos(vr_ok["prod"].tolist(), "VR<0.9 (均值回归友好)")
vr_bad = R[(R.vr >= 1.0)].dropna(subset=["vr"])
if len(vr_bad):
    basket_oos(vr_bad["prod"].tolist(), "VR>=1.0 (趋势/随机, 对照)")
# VR 是否预测样本外
vv = R.dropna(subset=["vr", "te_sharpe"])
print(f"\nVR 与 样本外夏普 的相关性: {vv['vr'].corr(vv['te_sharpe']):+.3f} "
      f"(应为负: VR越低越均值回归→样本外越好)")

R.sort_values("te_sharpe", ascending=False).to_csv(
    "data_futures/backtest_qiumo/product_screen.csv", index=False)
print("\n明细: data_futures/backtest_qiumo/product_screen.csv")
