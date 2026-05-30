"""品种结构画像: 为"按结构分型、分型配指标"做基础。

对每个品种计算结构特征(全~2.7年):
  vol_ann   日收益年化波动(衡量量级)
  atr_pct   中位 ATR/价 (单位波动)
  ER40      Kaufman效率比(40根≈1日): 高=趋势顺滑, 低=来回震荡
  VR10/VR40 方差比: <1 均值回归, >1 趋势/动量
  ac1       15min收益一阶自相关: <0 短期反转
  ac_abs    |收益|一阶自相关: 波动聚集强度
  skew/kurt 收益偏度/峰度

并输出:
  1) CS(玉米淀粉)的 收益率相关 Top 邻居 (经济联动)
  2) CS 的 结构特征相似 Top 邻居 (行为画像)
  3) 全品种按 ER/VR 粗分型
"""
import glob
import numpy as np
import pandas as pd
from download_futures import PRODUCTS

files = sorted(glob.glob("data_futures/15min/*.parquet"))


def er(c, n=40):
    d = np.abs(np.diff(c))
    sig = np.abs(c[n:] - c[:-n])
    noise = np.convolve(d, np.ones(n), "valid")          # 长度 len-n
    with np.errstate(divide="ignore", invalid="ignore"):
        e = sig / noise
    return np.nanmean(e[np.isfinite(e)])


def vr(r, q):
    r = r[~np.isnan(r)]
    m = len(r) // q * q
    if m < q * 5:
        return np.nan
    v1 = np.var(r)
    rq = np.add.reduceat(r[:m], np.arange(0, m, q))
    return (np.var(rq) / q) / v1 if v1 > 0 else np.nan


feats = []
dret = {}        # 日收益(用于相关)
for f in files:
    prod = f.split("/")[-1][:-8]
    df = pd.read_parquet(f).sort_values("datetime").reset_index(drop=True)
    c = df["close"].to_numpy()
    r = pd.Series(c).pct_change().to_numpy()
    # 日收益
    s = pd.Series(c, index=pd.to_datetime(df["datetime"]))
    dly = s.groupby(s.index.normalize()).last().pct_change()
    dret[prod] = dly
    H, L, C = df["high"].to_numpy(), df["low"].to_numpy(), c
    tr = np.maximum(H[1:] - L[1:], np.maximum(np.abs(H[1:] - C[:-1]),
                                              np.abs(L[1:] - C[:-1])))
    feats.append({
        "prod": prod, "name": PRODUCTS.get(prod, ("", prod))[1],
        "exch": PRODUCTS.get(prod, ("?",))[0],
        "vol_ann": dly.std() * np.sqrt(252),
        "atr_pct": np.nanmedian(tr / C[1:]),
        "ER40": er(c, 40),
        "VR10": vr(r, 10), "VR40": vr(r, 40),
        "ac1": pd.Series(r).autocorr(1),
        "ac_abs": pd.Series(np.abs(r)).autocorr(1),
        "skew": pd.Series(r).skew(), "kurt": pd.Series(r).kurt(),
    })

F = pd.DataFrame(feats).set_index("prod")
TARGET = "CS"

# 1) 收益率相关
D = pd.DataFrame(dret).dropna(how="all")
corr = D.corr().get(TARGET).drop(TARGET).sort_values(ascending=False)

# 2) 结构相似(标准化欧氏距离)
cols = ["vol_ann", "atr_pct", "ER40", "VR10", "VR40", "ac1", "ac_abs"]
Z = (F[cols] - F[cols].mean()) / F[cols].std()
dist = ((Z - Z.loc[TARGET]) ** 2).sum(axis=1).pow(0.5).drop(TARGET).sort_values()

pd.set_option("display.width", 200)
print("=" * 84)
print(f"目标品种: {TARGET} {F.loc[TARGET,'name']}  结构特征:")
print(F.loc[[TARGET], cols + ["skew", "kurt"]].round(3).to_string())
print("=" * 84)
print(f"1) 与 {TARGET} 收益率相关最高的 12 个品种 (经济联动):")
for p, v in corr.head(12).items():
    print(f"   {p:4s} {F.loc[p,'name']:<6} 相关={v:+.2f}  "
          f"(同所={F.loc[p,'exch']}, ER40={F.loc[p,'ER40']:.2f}, VR40={F.loc[p,'VR40']:.2f})")
print("=" * 84)
print(f"2) 与 {TARGET} 结构画像最相似的 12 个品种 (行为相近):")
for p, v in dist.head(12).items():
    print(f"   {p:4s} {F.loc[p,'name']:<6} 距离={v:.2f}  "
          f"vol={F.loc[p,'vol_ann']:.1%} ER40={F.loc[p,'ER40']:.2f} "
          f"VR40={F.loc[p,'VR40']:.2f} ac1={F.loc[p,'ac1']:+.3f}")
print("=" * 84)
print("3) 全品种粗分型 (按 ER40 趋势效率 × vol_ann 波动量级):")
F2 = F.copy()
F2["型"] = np.where(F2.ER40 >= F2.ER40.median(), "趋势型", "震荡型")
F2["波动"] = np.where(F2.vol_ann >= F2.vol_ann.median(), "高波", "低波")
for grp, sub in F2.groupby(["型", "波动"]):
    members = ", ".join(f"{i}({sub.loc[i,'name']})" for i in sub.index)
    print(f"\n[{grp[0]}·{grp[1]}] {len(sub)}个: {members}")

F.round(4).sort_values("ER40").to_csv("data_futures/backtest_qiumo/structure_profile.csv")
print("\n明细: data_futures/backtest_qiumo/structure_profile.csv")
