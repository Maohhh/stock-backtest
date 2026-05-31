"""① 农产品季节性深挖做第三条腿  ② 趋势突破加止损/多通道平均磨平尖峰。
真实成本2.5bp, 全程防前视。"""
import glob, numpy as np, pandas as pd
import strategy_range as sr, equity_curve as ec
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
def sh(s):s=s.dropna();return s.mean()/s.std()*np.sqrt(252) if s.std()>0 else 0
def fin(s,nm):
    s=s.dropna();eq=(1+s).cumprod();d=(s.index[-1]-s.index[0]).days;ann=eq.iloc[-1]**(365/d)-1
    segs=np.array_split(np.arange(len(s)),5);ss=[sh(s.iloc[ix]) for ix in segs]
    print(f"{nm:<22} 夏普={sh(s):.2f} 年化={ann:+.1%} 回撤={(eq/eq.cummax()-1).min():.1%} 5段=[{' '.join(f'{x:+.1f}' for x in ss)}]")

# ===== ① 季节性深挖 =====
print("="*78);print("① 农产品季节性 (留过去年均值防前视, 多种细化)");print("="*78)
AGRI=[p for p in ["A","B","M","Y","P","C","CS","RM","OI","SR","CF","AP","JD","CJ","PK","LH","UR"] if p in close.columns]
# 信号: 该品种该"日历月"在历史(仅更早年份)的平均日收益符号
def seas_sig(strength=False):
    sig=pd.DataFrame(0.0,index=ret.index,columns=AGRI)
    for p in AGRI:
        s=ret[p]
        for dt in ret.index:
            hist=s[(s.index.month==dt.month)&(s.index.year<dt.year)]
            if len(hist)>15:
                sig.loc[dt,p]=hist.mean() if strength else np.sign(hist.mean())
    return sig
reb=ret.index[::20];mask=pd.Series(ret.index.isin(reb),index=ret.index)
for nm,stg in [("符号等权",False),("强度加权",True)]:
    sig=seas_sig(stg)
    w=sig.div(sig.abs().sum(axis=1).replace(0,1),axis=0).where(mask,np.nan).ffill()
    pnl=(w.shift(1)*ret[AGRI]).sum(axis=1);turn=(w-w.shift(1)).abs().sum(axis=1)
    fin((pnl-turn.shift(1).fillna(0)*COST),f"  季节{nm}")
# 季节性日收益(用强度加权版做第三条腿)
sig=seas_sig(True);ws=sig.div(sig.abs().sum(axis=1).replace(0,1),axis=0).where(mask,np.nan).ffill()
seas=((ws.shift(1)*ret[AGRI]).sum(axis=1)-(ws-ws.shift(1)).abs().sum(axis=1).shift(1).fillna(0)*COST).dropna()

# ===== ② 趋势突破稳健化: 多通道平均(20/40/55/80) + ATR止损 =====
print("\n"+"="*78);print("② 趋势突破稳健化 (多通道平均, 磨平N=40尖峰)");print("="*78)
def trend_multi(Ns,stop=None):
    wsum=0
    for N in Ns:
        hh=close.rolling(N).max().shift(1);ll=close.rolling(N).min().shift(1)
        brk=np.where(close>hh,1,np.where(close<ll,-1,np.nan))
        wsum=wsum+pd.DataFrame(brk,index=close.index,columns=close.columns).ffill(limit=N).fillna(0)
    w=(wsum/len(Ns))/atr/close            # 平均信号, 波动率归一
    w=w.div(w.abs().sum(axis=1),axis=0)
    pnl=(w.shift(1)*ret).sum(axis=1);turn=(w-w.shift(1)).abs().sum(axis=1)
    return (pnl-turn.shift(1).fillna(0)*COST).dropna()
fin(trend_multi([40]),"  单通道N=40")
fin(trend_multi([20,40,55,80]),"  多通道平均(20/40/55/80)")
fin(trend_multi([30,40,55]),"  多通道平均(30/40/55)")
trd=trend_multi([20,40,55,80])

# 反转
CLEAN9=["V","PX","MA","SH","BU","RM","UR","PG","AP"]
rev=pd.DataFrame({p:ec.sim(sr.DATA[p],100,2.0,3.0) for p in CLEAN9}).fillna(0).mean(axis=1)

# ===== 三策略组合 =====
print("\n"+"="*78);print("三策略组合 (反转+趋势+季节, 各等波动)");print("="*78)
j=pd.concat([rev,trd,seas],axis=1,keys=["rev","trd","sea"]).dropna()
print("两两相关:")
print(j.corr().round(2).to_string())
rs=j["rev"]/j["rev"].std();ts=j["trd"]/j["trd"].std();ss_=j["sea"]/j["sea"].std()
for nm,wts in [("反转+趋势(50/50)",(0.5,0.5,0)),("三策略等权",(1/3,1/3,1/3)),
               ("反转40趋势40季节20",(0.4,0.4,0.2)),("反转45趋势45季节10",(0.45,0.45,0.1))]:
    cmb=(wts[0]*rs+wts[1]*ts+wts[2]*ss_)*j["rev"].std()
    fin(cmb,f"  {nm}")
