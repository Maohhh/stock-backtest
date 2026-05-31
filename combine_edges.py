"""验证趋势突破的稳健性 + 与反转策略的相关性 -> 能否组合。

核心: 反转(§10)和趋势突破若低相关, 组合可显著提升(两个独立edge)。
这正是回答'除了均值回归还有什么'——趋势突破是真实的第二类, 且互补。
"""
import glob, numpy as np, pandas as pd
import strategy_range as sr, equity_curve as ec
COST=0.00025
files=sorted(glob.glob("data_futures/15min/*.parquet"))
def dbars(f):
    df=pd.read_parquet(f);df["date"]=pd.to_datetime(df["datetime"]).dt.normalize()
    return df.groupby("date").agg(close=("close","last"),high=("high","max"),low=("low","min"))
D={f.split("/")[-1][:-8]:dbars(f) for f in files}
D={p:d for p,d in D.items() if len(d)>400}
close=pd.DataFrame({p:D[p]["close"] for p in D}).sort_index();ret=close.pct_change()
tr=pd.DataFrame({p:np.maximum(D[p]["high"]-D[p]["low"],np.maximum(
    (D[p]["high"]-D[p]["close"].shift()).abs(),(D[p]["low"]-D[p]["close"].shift()).abs())) for p in D})
atr=tr.rolling(20).mean()

# 趋势突破组合日收益(40日通道, 波动率归一, 含成本)
def trend_pnl(N=40):
    hh=close.rolling(N).max().shift(1);ll=close.rolling(N).min().shift(1)
    brk=np.where(close>hh,1,np.where(close<ll,-1,np.nan))
    w=pd.DataFrame(brk,index=close.index,columns=close.columns).ffill(limit=N)
    w=(w/atr/close); w=w.div(w.abs().sum(axis=1),axis=0)
    pnl=(w.shift(1)*ret).sum(axis=1);turn=(w-w.shift(1)).abs().sum(axis=1)
    return (pnl-turn.shift(1).fillna(0)*COST).dropna()

def sh(s):s=s.dropna();return s.mean()/s.std()*np.sqrt(252) if s.std()>0 else 0
def stats(s,nm):
    s=s.dropna();eq=(1+s).cumprod();days=(s.index[-1]-s.index[0]).days
    ann=eq.iloc[-1]**(365/days)-1;mdd=(eq/eq.cummax()-1).min()
    segs=np.array_split(np.arange(len(s)),5);ss=[sh(s.iloc[ix]) for ix in segs]
    print(f"{nm:<20} 夏普={sh(s):.2f} 年化={ann:+.1%} 回撤={mdd:.1%} Calmar={ann/abs(mdd):.2f} 5段最差={min(ss):+.1f}")

# 反转(9品种)组合日收益
CLEAN9=["V","PX","MA","SH","BU","RM","UR","PG","AP"]
rev=pd.DataFrame({p:ec.sim(sr.DATA[p],100,2.0,3.0) for p in CLEAN9}).fillna(0).mean(axis=1)
trd=trend_pnl(40)

print("两个独立策略:")
stats(rev,"反转(9品种)")
stats(trd,"趋势突破(40日全品种)")
# 相关性
j=pd.concat([rev,trd],axis=1,keys=["rev","trd"]).dropna()
c=j["rev"].corr(j["trd"])
print(f"\n两策略日收益相关性: {c:+.2f}  (<0或~0 => 强互补)")

# 等权组合(各半仓)
print("\n组合(反转50% + 趋势50%):")
# 先把两者各自缩放到同等波动再合
rev_s=rev/rev.std();trd_s=trd/trd.std()
combo=0.5*rev_s+0.5*trd_s
combo=combo*rev.std()  # 还原到反转量级便于看
stats(combo,"50/50组合")
# 风险平价合并(逆波动)
for wr in [0.3,0.5,0.7]:
    cmb=(wr*rev_s+(1-wr)*trd_s)
    print(f"  反转{wr:.0%}/趋势{1-wr:.0%}: 夏普={sh(cmb):.2f}")
