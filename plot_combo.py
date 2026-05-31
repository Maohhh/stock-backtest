import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import strategy_range as sr, equity_curve as ec
import glob
COST=0.00025
# 反转(9品种)
CLEAN9=["V","PX","MA","SH","BU","RM","UR","PG","AP"]
rev=pd.DataFrame({p:ec.sim(sr.DATA[p],100,2.0,3.0) for p in CLEAN9}).fillna(0).mean(axis=1)
# 趋势突破(40日全品种)
files=sorted(glob.glob("data_futures/15min/*.parquet"))
def dbars(f):
    df=pd.read_parquet(f);df["date"]=pd.to_datetime(df["datetime"]).dt.normalize()
    return df.groupby("date").agg(close=("close","last"),high=("high","max"),low=("low","min"))
D={f.split("/")[-1][:-8]:dbars(f) for f in files};D={p:d for p,d in D.items() if len(d)>400}
close=pd.DataFrame({p:D[p]["close"] for p in D}).sort_index();ret=close.pct_change()
tr=pd.DataFrame({p:np.maximum(D[p]["high"]-D[p]["low"],np.maximum(
    (D[p]["high"]-D[p]["close"].shift()).abs(),(D[p]["low"]-D[p]["close"].shift()).abs())) for p in D})
atr=tr.rolling(20).mean()
hh=close.rolling(40).max().shift(1);ll=close.rolling(40).min().shift(1)
brk=np.where(close>hh,1,np.where(close<ll,-1,np.nan))
w=pd.DataFrame(brk,index=close.index,columns=close.columns).ffill(limit=40);w=(w/atr/close);w=w.div(w.abs().sum(axis=1),axis=0)
trd=((w.shift(1)*ret).sum(axis=1)-(w-w.shift(1)).abs().sum(axis=1).shift(1).fillna(0)*COST).dropna()

j=pd.concat([rev,trd],axis=1,keys=["rev","trd"]).dropna()
# 等波动合并到 反转量级
rv=j["rev"]/j["rev"].std()*j["rev"].std()
rev_s=j["rev"]/j["rev"].std();trd_s=j["trd"]/j["trd"].std()
combo=(0.5*rev_s+0.5*trd_s)*j["rev"].std()
cap0=100000
def curve(s):return cap0*(1+s).cumprod()
fig,axes=plt.subplots(2,1,figsize=(12,8),gridspec_kw={"height_ratios":[3,1]},sharex=True)
for s,c,lab in [(j["rev"],"#1f77b4","反转(9品种) 夏普1.80/回撤-7.9%"),
                (j["trd"],"#ff7f0e","趋势突破(40日) 夏普1.09/回撤-18.9%"),
                (combo,"#2ca02c","50/50组合 夏普2.36/回撤-6.1%")]:
    axes[0].plot(curve(s).index,curve(s).values,color=c,lw=1.7,label=lab)
axes[0].axhline(cap0,color="gray",ls="--",lw=0.8)
axes[0].set_yscale("log");axes[0].set_ylabel("资金(元,对数轴)")
axes[0].set_title("反转 vs 趋势突破 vs 组合 · 10万本金 (真实成本2.5bp, 2023-09~2026-05)")
axes[0].legend(loc="upper left",fontsize=9);axes[0].grid(True,alpha=0.3)
eqc=(1+combo).cumprod();dd=(eqc/eqc.cummax()-1)
axes[1].fill_between(dd.index,dd.values*100,0,color="#2ca02c",alpha=0.4)
axes[1].set_ylabel("组合回撤%");axes[1].set_xlabel("日期");axes[1].grid(True,alpha=0.3)
plt.tight_layout();plt.savefig("data_futures/backtest_qiumo/combo_equity.png",dpi=110)
def sh(s):return s.mean()/s.std()*np.sqrt(252)
def fin(s,nm):
    eq=(1+s).cumprod();d=(s.index[-1]-s.index[0]).days;ann=eq.iloc[-1]**(365/d)-1
    print(f"{nm}: 10万->{cap0*eq.iloc[-1]:,.0f}元 年化{ann:+.1%} 夏普{sh(s):.2f} 回撤{(eq/eq.cummax()-1).min():.1%}")
print("saved combo_equity.png")
fin(j["rev"],"反转 ");fin(j["trd"],"趋势 ");fin(combo,"组合 ")
