import glob, numpy as np, pandas as pd
import factor_macd2 as fm
COST=0.00025
files=sorted(glob.glob("data_futures/15min/*.parquet"))
# MACD(带趋势过滤)组合日收益
D={}
for f in files:
    df=pd.read_parquet(f).sort_values("datetime").reset_index(drop=True)
    if len(df)>2000:D[f.split('/')[-1][:-8]]=df
macd_pnl=pd.DataFrame({p:fm.run(D[p],True,False) for p in D}).fillna(0).mean(axis=1)
# 趋势突破(Donchian40)
def dbars(f):
    df=pd.read_parquet(f);df["date"]=pd.to_datetime(df["datetime"]).dt.normalize()
    return df.groupby("date").agg(close=("close","last"),high=("high","max"),low=("low","min"))
Dd={f.split("/")[-1][:-8]:dbars(f) for f in files};Dd={p:d for p,d in Dd.items() if len(d)>400}
close=pd.DataFrame({p:Dd[p]["close"] for p in Dd}).sort_index();ret=close.pct_change()
tr=pd.DataFrame({p:np.maximum(Dd[p]["high"]-Dd[p]["low"],np.maximum((Dd[p]["high"]-Dd[p]["close"].shift()).abs(),(Dd[p]["low"]-Dd[p]["close"].shift()).abs())) for p in Dd})
atr=tr.rolling(20).mean()
hh=close.rolling(40).max().shift(1);ll=close.rolling(40).min().shift(1)
brk=np.where(close>hh,1,np.where(close<ll,-1,np.nan))
w=pd.DataFrame(brk,index=close.index,columns=close.columns).ffill(limit=40);w=(w/atr/close);w=w.div(w.abs().sum(axis=1),axis=0)
trd=((w.shift(1)*ret).sum(axis=1)-(w-w.shift(1)).abs().sum(axis=1).shift(1).fillna(0)*COST)
j=pd.concat([macd_pnl,trd],axis=1,keys=["macd","trend"]).dropna()
print(f"MACD(带趋势过滤) vs Donchian趋势突破 相关性: {j['macd'].corr(j['trend']):+.2f}")
print(f"MACD夏普={j['macd'].mean()/j['macd'].std()*np.sqrt(252):.2f}  趋势突破夏普={j['trend'].mean()/j['trend'].std()*np.sqrt(252):.2f}")
