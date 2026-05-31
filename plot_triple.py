import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import strategy_range as sr, equity_curve as ec, glob
COST=0.00025
files=sorted(glob.glob("data_futures/15min/*.parquet"))
def dbars(f):
    df=pd.read_parquet(f);df["date"]=pd.to_datetime(df["datetime"]).dt.normalize()
    return df.groupby("date").agg(close=("close","last"),high=("high","max"),low=("low","min"))
D={f.split("/")[-1][:-8]:dbars(f) for f in files};D={p:d for p,d in D.items() if len(d)>400}
close=pd.DataFrame({p:D[p]["close"] for p in D}).sort_index();ret=close.pct_change()
tr=pd.DataFrame({p:np.maximum(D[p]["high"]-D[p]["low"],np.maximum(
    (D[p]["high"]-D[p]["close"].shift()).abs(),(D[p]["low"]-D[p]["close"].shift()).abs())) for p in D})
atr=tr.rolling(20).mean()
CLEAN9=["V","PX","MA","SH","BU","RM","UR","PG","AP"]
rev=pd.DataFrame({p:ec.sim(sr.DATA[p],100,2.0,3.0) for p in CLEAN9}).fillna(0).mean(axis=1)
hh=close.rolling(40).max().shift(1);ll=close.rolling(40).min().shift(1)
brk=np.where(close>hh,1,np.where(close<ll,-1,np.nan))
w=pd.DataFrame(brk,index=close.index,columns=close.columns).ffill(limit=40);w=(w/atr/close);w=w.div(w.abs().sum(axis=1),axis=0)
trd=((w.shift(1)*ret).sum(axis=1)-(w-w.shift(1)).abs().sum(axis=1).shift(1).fillna(0)*COST).dropna()
AGRI=[p for p in ["A","B","M","Y","P","C","CS","RM","OI","SR","CF","AP","JD","CJ","PK","LH","UR"] if p in close.columns]
sig=pd.DataFrame(0.0,index=ret.index,columns=AGRI)
for p in AGRI:
    s=ret[p]
    for dt in ret.index:
        h=s[(s.index.month==dt.month)&(s.index.year<dt.year)]
        if len(h)>15: sig.loc[dt,p]=h.mean()
reb=ret.index[::20];mask=pd.Series(ret.index.isin(reb),index=ret.index)
wsg=sig.div(sig.abs().sum(axis=1).replace(0,1),axis=0).where(mask,np.nan).ffill()
sea=((wsg.shift(1)*ret[AGRI]).sum(axis=1)-(wsg-wsg.shift(1)).abs().sum(axis=1).shift(1).fillna(0)*COST).dropna()
j=pd.concat([rev,trd,sea],axis=1,keys=["r","t","s"]).dropna()
rs=j["r"]/j["r"].std();ts=j["t"]/j["t"].std();ss=j["s"]/j["s"].std()
combo=((rs+ts+ss)/3)*j["r"].std()
cap0=100000;cur=lambda s:cap0*(1+s).cumprod()
fig,ax=plt.subplots(2,1,figsize=(12,8),gridspec_kw={"height_ratios":[3,1]},sharex=True)
for s,c,l in [(j["r"],"#1f77b4","反转 夏普1.8/回撤-7.9%"),(j["t"],"#ff7f0e","趋势突破 夏普1.1/回撤-18.9%"),
              (j["s"],"#9467bd","农产品季节性 夏普0.65"),(combo,"#2ca02c","三策略等权 夏普2.09/回撤-3.4%")]:
    ax[0].plot(cur(s).index,cur(s).values,color=c,lw=1.7,label=l)
ax[0].axhline(cap0,color="gray",ls="--",lw=0.8);ax[0].set_yscale("log")
ax[0].set_ylabel("资金(元,对数轴)");ax[0].legend(loc="upper left",fontsize=8.5);ax[0].grid(True,alpha=0.3)
ax[0].set_title("三策略组合 · 10万本金 (反转+趋势突破+农产品季节性, 真实成本2.5bp)")
eqc=(1+combo).cumprod();dd=(eqc/eqc.cummax()-1)
ax[1].fill_between(dd.index,dd.values*100,0,color="#2ca02c",alpha=0.4)
ax[1].set_ylabel("组合回撤%");ax[1].grid(True,alpha=0.3)
plt.tight_layout();plt.savefig("data_futures/backtest_qiumo/triple_equity.png",dpi=110)
eq=(1+combo).cumprod();d=(combo.index[-1]-combo.index[0]).days
print(f"三策略等权: 10万->{cap0*eq.iloc[-1]:,.0f}元 年化{eq.iloc[-1]**(365/d)-1:+.1%} 夏普{combo.mean()/combo.std()*np.sqrt(252):.2f} 回撤{(eq/eq.cummax()-1).min():.1%}")
print("saved triple_equity.png")
