"""可交易版: 在'回归型(VR低)'基础上加流动性过滤, 干净集上 walk-forward + 成本压测。

流动性: 各品种 中位(volume*close) 作为日内成交额代理; 剔除最低档(垃圾/低流动)。
另剔除极端波动毛刺(年化vol>60%, 如胶合板)。
"""
import numpy as np, pandas as pd
import strategy_range as sr
from download_futures import PRODUCTS

prof = sr.prof
# 流动性代理
liq = {}
for p, df in sr.DATA.items():
    v = (df["volume"] * df["close"]).median()
    liq[p] = v
L = pd.Series(liq)

def sim(df, N=100, K=2.0, stop_atr=3.0, cost=0.0005):
    C=df["close"];H=df["high"];Lo=df["low"]
    mid=C.rolling(N,min_periods=N).mean();sd=C.rolling(N,min_periods=N).std()
    up=(mid+K*sd).to_numpy();lo=(mid-K*sd).to_numpy();midn=mid.to_numpy()
    c=C.to_numpy();h=H.to_numpy();l=Lo.to_numpy()
    tr=pd.concat([H-Lo,(H-C.shift()).abs(),(Lo-C.shift()).abs()],axis=1).max(axis=1)
    atr=tr.rolling(sr.ATR_N,min_periods=sr.ATR_N).mean().to_numpy()
    n=len(c);warm=max(N,sr.ATR_N)+1;rb=np.zeros(n);p=0;ei=-1;es=0.0
    for i in range(warm,n):
        if np.isnan(midn[i]) or np.isnan(atr[i-1]): continue
        ret=0.0
        if p!=0:
            px=c[i];ex=False
            if p>0:
                if l[i]<=es:px=es;ex=True
                elif c[i]>=midn[i] or (i-ei)>=sr.MAXH:ex=True
            else:
                if h[i]>=es:px=es;ex=True
                elif c[i]<=midn[i] or (i-ei)>=sr.MAXH:ex=True
            ret=p*(px/c[i-1]-1)
            if ex:ret-=cost;p=0
        rb[i]=ret
        if p==0:
            if c[i]<lo[i]:p=1;ei=i;es=c[i]-stop_atr*atr[i];rb[i]-=cost
            elif c[i]>up[i]:p=-1;ei=i;es=c[i]+stop_atr*atr[i];rb[i]-=cost
    s=pd.Series(rb,index=pd.to_datetime(df["datetime"]))
    return s.groupby(s.index.normalize()).sum()

def wf(prods, cost):
    cols={p:sim(sr.DATA[p],cost=cost) for p in prods}
    port=pd.DataFrame(cols).fillna(0).mean(axis=1).sort_index().dropna()
    segs=np.array_split(np.arange(len(port)),5)
    ss=[port.iloc[ix].mean()/port.iloc[ix].std()*np.sqrt(252) if port.iloc[ix].std()>0 else 0 for ix in segs]
    full=port.mean()/port.std()*np.sqrt(252)
    days=(port.index[-1]-port.index[0]).days
    ann=(1+port).prod()**(365/days)-1
    mdd=((1+port).cumprod()/(1+port).cumprod().cummax()-1).min()
    return full,ss,ann,mdd

# 回归型候选 = VR40<0.8
vr=prof["VR40"].dropna()
cand=vr[vr<0.8].index.tolist()
print(f"回归型候选(VR40<0.8): {len(cand)}个")
# 流动性分档
Lc=L[cand].sort_values()
print("\n按流动性(中位 volume*close 代理)排序, 低->高:")
for p in Lc.index:
    flag=""
    if prof.loc[p,'vol_ann']>0.6: flag="⚠超高波"
    print(f"  {p:4s} {PRODUCTS.get(p,('',p))[1]:<6} 流动性={Lc[p]:>14,.0f}  vol={prof.loc[p,'vol_ann']:.0%} VR={prof.loc[p,'VR40']:.2f} {flag}")

# 过滤规则: 剔除流动性最低30% + 年化波动>60%
liq_thresh=Lc.quantile(0.30)
clean=[p for p in cand if L[p]>=liq_thresh and prof.loc[p,'vol_ann']<=0.60]
dropped=[p for p in cand if p not in clean]
print(f"\n过滤(剔流动性最低30% & vol>60%): 保留{len(clean)}, 剔除{len(dropped)}")
print(f"剔除: {' '.join(dropped)}")
print(f"保留: {' '.join(clean)}")

print("\n"+"="*78)
print("walk-forward 对比 (5段夏普, 成本5bp/边):")
print("="*78)
for label,prods in [("全候选(VR<0.8,含垃圾)",cand),("干净集(过滤后)",clean)]:
    full,ss,ann,mdd=wf(prods,0.0005)
    print(f"{label:<22} 全期夏普={full:>5.2f} 年化={ann:>+6.1%} 回撤={mdd:>6.1%} "
          f"5段=[{' '.join(f'{s:+.1f}' for s in ss)}] 最差={min(ss):+.1f}")
print("\n干净集 成本敏感性:")
for cost in [0.0005,0.001,0.0015,0.002]:
    full,ss,ann,mdd=wf(clean,cost)
    print(f"  {cost*1e4:>4.0f}bp/边: 全期夏普={full:>5.2f} 年化={ann:>+6.1%} 5段最差={min(ss):+.1f}")

# ---- 修正: 用成交手数(volume)的中位 + 非零bar占比 作流动性, 不受价格/乘数失真 ----
print("\n"+"="*78); print("修正流动性度量(成交手数中位 + 活跃bar占比)"); print("="*78)
liq2={}; active={}
for p,df in sr.DATA.items():
    liq2[p]=df["volume"].median()
    active[p]=(df["volume"]>0).mean()
L2=pd.Series(liq2); AC=pd.Series(active)
vr=prof["VR40"].dropna(); cand=vr[vr<0.8].index.tolist()
# 干净规则: 成交手数中位>=2000手 且 活跃bar>95% 且 0.08<=vol<=0.6 (剔死品种/超高波/没波动)
clean2=[p for p in cand if L2[p]>=2000 and AC[p]>=0.95
        and 0.08<=prof.loc[p,'vol_ann']<=0.60]
dropped2=[p for p in cand if p not in clean2]
print(f"保留{len(clean2)}: {' '.join(clean2)}")
print(f"剔除{len(dropped2)}: {' '.join(dropped2)}")
for p in clean2:
    print(f"  {p:4s} {PRODUCTS.get(p,('',p))[1]:<6} 手数中位={L2[p]:>8,.0f} 活跃={AC[p]:.0%} vol={prof.loc[p,'vol_ann']:.0%} VR={prof.loc[p,'VR40']:.2f}")
print("\nwalk-forward(成本5/10bp):")
for cost in [0.0005,0.001]:
    full,ss,ann,mdd=wf(clean2,cost)
    print(f"  {cost*1e4:>4.0f}bp: 全期夏普={full:>5.2f} 年化={ann:>+6.1%} 回撤={mdd:>6.1%} 5段=[{' '.join(f'{s:+.1f}' for s in ss)}] 最差={min(ss):+.1f}")
