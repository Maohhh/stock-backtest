"""多周期共振版MACD (朋友的真正方法论): 大周期定方向, 小周期定入场。

级别(由15min聚合): 日线(大) ~ 60min(中) ~ 15min(小)
规则(标准多周期共振):
  大周期(日线) MACD DIF>0 (零轴上=多头大趋势) -> 只允许做多
  中周期(60min) DIF>0 确认
  小周期(15min) MA10金叉MA20 且 DIF>0 -> 入场做多
  空头对称。出场: 小周期反向死叉 / 大周期趋势消失(DIF翻负)
真实成本2.5bp, 全品种, walk-forward。多种级别组合对比。
"""
import glob, numpy as np, pandas as pd
COST=0.00025
files=sorted(glob.glob("data_futures/15min/*.parquet"))
def macd(c,f=12,s=26,sig=9):
    dif=c.ewm(span=f,adjust=False).mean()-c.ewm(span=s,adjust=False).mean()
    return dif, dif.ewm(span=sig,adjust=False).mean()
def sh(s):s=s.dropna();return s.mean()/s.std()*np.sqrt(252) if s.std()>0 else 0
def report(port,nm):
    port=port.dropna()
    if len(port)<50:print(f"{nm}: 数据不足");return None
    eq=(1+port).cumprod();d=(port.index[-1]-port.index[0]).days;ann=eq.iloc[-1]**(365/d)-1
    mdd=(eq/eq.cummax()-1).min();segs=np.array_split(np.arange(len(port)),5);ss=[sh(port.iloc[ix]) for ix in segs]
    print(f"{nm:<34} 夏普={sh(port):>5.2f} 年化={ann:>+6.1%} 回撤={mdd:>6.1%} 5段=[{' '.join(f'{x:+.1f}' for x in ss)}]")
    return port

def agg(df, bars):
    """把15min聚合成 bars*15min 的更大周期K线。"""
    g=df.iloc[::1].copy()
    g["grp"]=np.arange(len(g))//bars
    o=g.groupby("grp").agg(datetime=("datetime","last"),close=("close","last"))
    return o.set_index("datetime")["close"]

def run_mtf(df, big_tf, mid_tf=None):
    """big_tf/mid_tf: 多少根15min合成(日线≈16根, 60min=4根, 15min=1)。
    大周期MACD零轴定方向, (可选中周期确认), 小周期(15min)MA金叉入场。"""
    c15=df["close"].reset_index(drop=True)
    dt=pd.to_datetime(df["datetime"]).reset_index(drop=True)
    # 大周期DIF, 对齐回15min(用ffill, 防前视: 大周期K线收盘后才可用)
    big=agg(df,big_tf); bdif,bdea=macd(big)
    big_dir=pd.Series(np.where(bdif>0,1,np.where(bdif<0,-1,0)),index=big.index)
    big_al=big_dir.reindex(dt,method="ffill").to_numpy()
    if mid_tf:
        mid=agg(df,mid_tf);mdif,_=macd(mid)
        mid_dir=pd.Series(np.where(mdif>0,1,np.where(mdif<0,-1,0)),index=mid.index)
        mid_al=mid_dir.reindex(dt,method="ffill").to_numpy()
    else:
        mid_al=np.zeros(len(dt))+99  # 不约束
    # 小周期15min MA金叉 + 自身MACD
    ma10=c15.rolling(10).mean();ma20=c15.rolling(20).mean()
    dif15,_=macd(c15)
    gc=((ma10>ma20)&(ma10.shift(1)<=ma20.shift(1))).to_numpy()
    dc=((ma10<ma20)&(ma10.shift(1)>=ma20.shift(1))).to_numpy()
    cl=c15.to_numpy();n=len(cl);pos=np.zeros(n);p=0
    d15=dif15.to_numpy()
    for i in range(20,n):
        long_ok = big_al[i]>0 and (mid_al[i]>0 or mid_al[i]==99) and d15[i]>0
        short_ok= big_al[i]<0 and (mid_al[i]<0 or mid_al[i]==99) and d15[i]<0
        if gc[i] and long_ok: p=1
        elif dc[i] and short_ok: p=-1
        elif p>0 and (dc[i] or big_al[i]<=0): p=0      # 反向死叉或大趋势消失
        elif p<0 and (gc[i] or big_al[i]>=0): p=0
        pos[i]=p
    ret=np.zeros(n);ret[1:]=pos[:-1]*(cl[1:]/cl[:-1]-1);ret-=np.abs(np.diff(pos,prepend=0))*COST
    s=pd.Series(ret,index=dt);return s.groupby(s.index.normalize()).sum()

D={}
for f in files:
    df=pd.read_parquet(f).sort_values("datetime").reset_index(drop=True)
    if len(df)>3000: D[f.split('/')[-1][:-8]]=df
print(f"{len(D)}品种 (日线≈16根15min, 60min=4根)")
# 对照: 单周期(上一节,亏) vs 多周期共振
for nm,big,mid in [("仅15min(对照,单周期)",1,None),
                   ("日线定向+15min入场",16,None),
                   ("日线+60min+15min三周期共振",16,4),
                   ("60min定向+15min入场",4,None)]:
    cols={p:run_mtf(D[p],big,mid) for p in D}
    report(pd.DataFrame(cols).fillna(0).mean(axis=1),nm)

print("\n=== 决定性对比: 多周期框架下, 入场方式 MA金叉 vs 通道突破 ===")
def run_mtf_breakout(df, big_tf, entry_n=20):
    """大周期(日线)MACD定方向 + 小周期(15min)Donchian通道突破入场(替代MA金叉)。"""
    c15=df["close"].reset_index(drop=True)
    dt=pd.to_datetime(df["datetime"]).reset_index(drop=True)
    big=agg(df,big_tf);bdif,_=macd(big)
    big_dir=pd.Series(np.where(bdif>0,1,np.where(bdif<0,-1,0)),index=big.index)
    big_al=big_dir.reindex(dt,method="ffill").to_numpy()
    cl=c15.to_numpy()
    hh=pd.Series(cl).rolling(entry_n).max().shift(1).to_numpy()
    ll=pd.Series(cl).rolling(entry_n).min().shift(1).to_numpy()
    n=len(cl);pos=np.zeros(n);p=0
    for i in range(entry_n+1,n):
        if big_al[i]>0 and cl[i]>hh[i]: p=1
        elif big_al[i]<0 and cl[i]<ll[i]: p=-1
        elif p>0 and (cl[i]<ll[i] or big_al[i]<=0): p=0
        elif p<0 and (cl[i]>hh[i] or big_al[i]>=0): p=0
        pos[i]=p
    ret=np.zeros(n);ret[1:]=pos[:-1]*(cl[1:]/cl[:-1]-1);ret-=np.abs(np.diff(pos,prepend=0))*COST
    s=pd.Series(ret,index=dt);return s.groupby(s.index.normalize()).sum()

for nm,en in [("日线定向+15min通道突破(N20)",20),("日线定向+15min通道突破(N40)",40)]:
    cols={p:run_mtf_breakout(D[p],16,en) for p in D}
    report(pd.DataFrame(cols).fillna(0).mean(axis=1),nm)
