"""给MACD系统最公平的机会: 加朋友强调的约束。
①大级别趋势过滤(慢线/长均线方向) ②降频(只在强信号) ③对比纯多头(商品有做多偏好?)
④背离的客观代理: 价格新低但MACD未新低=底背离。
15min真实成本2.5bp, walk-forward。"""
import glob, numpy as np, pandas as pd
COST=0.00025
files=sorted(glob.glob("data_futures/15min/*.parquet"))
def macd(c,f=12,s=26,sig=9):
    dif=c.ewm(span=f,adjust=False).mean()-c.ewm(span=s,adjust=False).mean()
    return dif, dif.ewm(span=sig,adjust=False).mean()
def sh(s):s=s.dropna();return s.mean()/s.std()*np.sqrt(252) if s.std()>0 else 0
def report(port,nm):
    port=port.dropna()
    if len(port)<50:print(f"{nm}:数据不足");return
    eq=(1+port).cumprod();d=(port.index[-1]-port.index[0]).days;ann=eq.iloc[-1]**(365/d)-1
    mdd=(eq/eq.cummax()-1).min();segs=np.array_split(np.arange(len(port)),5);ss=[sh(port.iloc[ix]) for ix in segs]
    print(f"{nm:<30} 夏普={sh(port):>5.2f} 年化={ann:>+6.1%} 回撤={mdd:>6.1%} 5段=[{' '.join(f'{x:+.1f}' for x in ss)}]")

def run(df, macro_filter=True, divergence=False):
    c=df["close"];dif,dea=macd(c)
    ma10=c.rolling(10).mean();ma20=c.rolling(20).mean()
    ma120=c.rolling(120).mean()  # 大级别趋势代理
    gc=(ma10>ma20)&(ma10.shift(1)<=ma20.shift(1))
    dc=(ma10<ma20)&(ma10.shift(1)>=ma20.shift(1))
    long_ok=dif>0; short_ok=dif<0
    if macro_filter:
        long_ok=long_ok&(c>ma120); short_ok=short_ok&(c<ma120)
    if divergence:
        # 底背离代理: 价格20根新低 但 DIF 未创新低
        pl=c==c.rolling(20).min(); difup=dif>dif.rolling(20).min()+1e-9
        ph=c==c.rolling(20).max(); difdn=dif<dif.rolling(20).max()-1e-9
        long_ok=long_ok&pl&difup; short_ok=short_ok&ph&difdn
    longsig=gc&long_ok; shortsig=dc&short_ok
    ls=longsig.to_numpy();ss=shortsig.to_numpy();cl=c.to_numpy()
    dcn=dc.to_numpy();gcn=gc.to_numpy();n=len(cl);pos=np.zeros(n);p=0
    for i in range(130,n):
        if ls[i]:p=1
        elif ss[i]:p=-1
        elif (p>0 and dcn[i]) or (p<0 and gcn[i]):p=0
        pos[i]=p
    ret=np.zeros(n);ret[1:]=pos[:-1]*(cl[1:]/cl[:-1]-1);ret-=np.abs(np.diff(pos,prepend=0))*COST
    s=pd.Series(ret,index=pd.to_datetime(df["datetime"]));return s.groupby(s.index.normalize()).sum()

D={}
for f in files:
    df=pd.read_parquet(f).sort_values("datetime").reset_index(drop=True)
    if len(df)>2000:D[f.split('/')[-1][:-8]]=df
SKIP={"BU","FG","I"}  # 朋友说不做的: 沥青/玻璃/铁矿
Dgood={p:d for p,d in D.items() if p not in SKIP}
print(f"全品种{len(D)}, 剔除沥青玻璃铁矿后{len(Dgood)}")
for nm,dd,mf,dv in [("①裸金叉(无过滤)",D,False,False),
                    ("②+大级别趋势过滤",D,True,False),
                    ("③+趋势+剔除不流畅品种",Dgood,True,False),
                    ("④+趋势+背离代理",D,True,True)]:
    cols={p:run(dd[p],mf,dv) for p in dd}
    report(pd.DataFrame(cols).fillna(0).mean(axis=1),nm)
