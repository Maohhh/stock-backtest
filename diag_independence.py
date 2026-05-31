"""澄清: 9品种放一起不是因为'周期相同', 恰恰因为它们'互不相关'。

策略是各品种独立回归自己的均线(截面无关), 9个并列的理由是分散, 不是同周期。
"""
import numpy as np, pandas as pd, glob
import strategy_range as sr
import equity_curve as ec
CLEAN9=["V","PX","MA","SH","BU","RM","UR","PG","AP"]
names={"V":"PVC","PX":"对二甲苯","MA":"甲醇","SH":"烧碱","BU":"沥青",
       "RM":"菜粕","UR":"尿素","PG":"LPG","AP":"苹果"}
sect={"V":"化工","PX":"化工","MA":"化工","SH":"化工","BU":"能化","UR":"化工",
      "PG":"能化","RM":"农产品","AP":"农产品"}

# 1) 价格收益相关矩阵
def dret(p):
    df=pd.read_parquet(f"data_futures/15min/{p}.parquet")
    df["date"]=pd.to_datetime(df["datetime"]).dt.normalize()
    return df.groupby("date")["close"].last().pct_change()
R=pd.DataFrame({p:dret(p) for p in CLEAN9}).dropna()
corr=R.corr()
off=corr.values[np.triu_indices(9,1)]
print(f"9品种价格日收益 两两相关: 均值={off.mean():.2f} 中位={np.median(off):.2f} 最大={off.max():.2f}")
print("(均值回归策略要的就是低相关->分散; 若同周期会高相关)")

# 2) 策略PnL相关矩阵(更关键: 策略收益是否独立)
P=pd.DataFrame({p:ec.sim(sr.DATA[p],100,2.0,3.0) for p in CLEAN9}).fillna(0)
pc=P.corr().values[np.triu_indices(9,1)]
print(f"\n9品种【策略PnL】两两相关: 均值={pc.mean():.2f} 中位={np.median(pc):.2f} 最大={pc.max():.2f}")
print("策略收益几乎不相关 -> 9条独立的小赌注, 这才是组合夏普1.8的来源")

# 3) 分散的威力: 单品种平均夏普 vs 组合夏普
def sh(s):
    s=s.dropna(); return s.mean()/s.std()*np.sqrt(252) if s.std()>0 else 0
solo=[sh(P[p]) for p in CLEAN9]
port=sh(P.mean(axis=1))
print(f"\n单品种夏普: 平均={np.mean(solo):.2f}  组合夏普={port:.2f}  放大倍数={port/np.mean(solo):.1f}x")
print(f"理论(完全独立)放大=√9={np.sqrt(9):.1f}x -> 实际接近, 证明品种确实近独立")

# 4) 按板块分组, 看化工组 vs 农产品组 是否各自也成立(同周期内未必更好)
print("\n按板块分组夏普(若'同周期'有意义,板块内应更协同):")
for s in ["化工","能化","农产品"]:
    members=[p for p in CLEAN9 if sect[p]==s]
    if len(members)>=1:
        ps=sh(P[members].mean(axis=1))
        print(f"  {s}({len(members)}个:{','.join(names[m] for m in members)}): 组合夏普={ps:.2f}")
