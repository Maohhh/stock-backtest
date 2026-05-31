"""验证'MACD零轴+MA10/20金叉'规则 (朋友说日内年化35%+)。

规则: MACD(12,26,9)
  做多: DIF>0(零轴上) 且 MA10上穿MA20
  做空: DIF<0(零轴下) 且 MA10下穿MA20
出场: 反向信号 / 日内版=当日收盘平
在15min K线(符合"最少5分钟级别"), 真实成本2.5bp/边, 全品种, walk-forward。
"""
import glob, numpy as np, pandas as pd
COST=0.00025
files=sorted(glob.glob("data_futures/15min/*.parquet"))

def macd(c, f=12, s=26, sig=9):
    ef=c.ewm(span=f,adjust=False).mean(); es=c.ewm(span=s,adjust=False).mean()
    dif=ef-es; dea=dif.ewm(span=sig,adjust=False).mean()
    return dif, dea

def run_one(df, intraday=False):
    c=df["close"]
    dif,dea=macd(c)
    ma10=c.rolling(10).mean(); ma20=c.rolling(20).mean()
    gc=(ma10>ma20)&(ma10.shift(1)<=ma20.shift(1))   # 金叉
    dc=(ma10<ma20)&(ma10.shift(1)>=ma20.shift(1))   # 死叉
    longsig = gc & (dif>0)
    shortsig= dc & (dif<0)
    ls=longsig.to_numpy(); ss=shortsig.to_numpy(); cl=c.to_numpy()
    n=len(cl); pos=np.zeros(n); p=0
    day=pd.to_datetime(df["datetime"]).dt.normalize().to_numpy()
    for i in range(30,n):
        if intraday and i>0 and day[i]!=day[i-1]: p=0   # 隔日平仓
        if ls[i]: p=1
        elif ss[i]: p=-1
        elif (p>0 and dc.iloc[i]) or (p<0 and gc.iloc[i]): p=0  # 反向均线交叉离场
        pos[i]=p
    ret=np.zeros(n); ret[1:]=pos[:-1]*(cl[1:]/cl[:-1]-1)
    turn=np.abs(np.diff(pos,prepend=0)); ret-=turn*COST
    s=pd.Series(ret,index=pd.to_datetime(df["datetime"]))
    return s.groupby(s.index.normalize()).sum()

def sh(s):s=s.dropna();return s.mean()/s.std()*np.sqrt(252) if s.std()>0 else 0
def report(port,nm):
    port=port.dropna();eq=(1+port).cumprod();d=(port.index[-1]-port.index[0]).days
    ann=eq.iloc[-1]**(365/d)-1;mdd=(eq/eq.cummax()-1).min()
    segs=np.array_split(np.arange(len(port)),5);ss=[sh(port.iloc[ix]) for ix in segs]
    print(f"{nm:<22} 夏普={sh(port):>5.2f} 年化={ann:>+6.1%} 回撤={mdd:>6.1%} 5段=[{' '.join(f'{x:+.1f}' for x in ss)}]")

D={}
for f in files:
    df=pd.read_parquet(f).sort_values("datetime").reset_index(drop=True)
    if len(df)>2000: D[f.split('/')[-1][:-8]]=df
print(f"{len(D)}品种, 15min")
for mode,lab in [(False,"持仓版(持到反向)"),(True,"日内版(隔夜平)")]:
    cols={p:run_one(D[p],mode) for p in D}
    port=pd.DataFrame(cols).fillna(0).mean(axis=1)
    report(port,f"全品种等权·{lab}")
