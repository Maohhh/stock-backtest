"""把'干净集区间反转'雏形做扎实(真实成本2.5bp/边)。

1) 参数稳健性: N(中枢)×K(带宽)×stop 网格, 看夏普是否平滑高原(非脆尖峰)
2) 逐品种 standalone 夏普 + 交易数, 区分真edge vs 搭便车
3) 逐品种 时间前后半样本 一致性(过拟合检验)
4) 波动率目标定仓的组合资金曲线(年化/夏普/回撤/Calmar)
"""
import glob, numpy as np, pandas as pd
import strategy_range as sr

COST = 0.00025            # 真实中性成本 2.5bp/边
CLEAN = ["V","PX","MA","SI","ZN","SH","BU","RM","UR","PG","AP"]

def sim(df, N, K, stop_atr, cost=COST):
    C=df["close"];H=df["high"];L=df["low"]
    mid=C.rolling(N,min_periods=N).mean();sd=C.rolling(N,min_periods=N).std()
    up=(mid+K*sd).to_numpy();lo=(mid-K*sd).to_numpy();midn=mid.to_numpy()
    c=C.to_numpy();h=H.to_numpy();l=L.to_numpy()
    tr=pd.concat([H-L,(H-C.shift()).abs(),(L-C.shift()).abs()],axis=1).max(axis=1)
    atr=tr.rolling(sr.ATR_N,min_periods=sr.ATR_N).mean().to_numpy()
    n=len(c);warm=max(N,sr.ATR_N)+1;rb=np.zeros(n);p=0;ei=-1;es=0.0;ntr=0
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
            if ex:ret-=cost;p=0;ntr+=1
        rb[i]=ret
        if p==0:
            if c[i]<lo[i]:p=1;ei=i;es=c[i]-stop_atr*atr[i];rb[i]-=cost
            elif c[i]>up[i]:p=-1;ei=i;es=c[i]+stop_atr*atr[i];rb[i]-=cost
    s=pd.Series(rb,index=pd.to_datetime(df["datetime"]))
    return s.groupby(s.index.normalize()).sum(), ntr

def sharpe(r):
    r=r.dropna()
    return r.mean()/r.std()*np.sqrt(252) if (len(r)>20 and r.std()>0) else 0.0

# 预存各品种日收益
DLY={p:sim(sr.DATA[p],100,2.0,3.0)[0] for p in CLEAN}

print("="*78)
print("1) 参数稳健性网格 (真实成本2.5bp, 组合夏普) — 应为平滑高原")
print("="*78)
print(f"{'':6}"+"".join(f"K={k:<6}" for k in [1.5,2.0,2.5,3.0]))
for N in [50,80,120,200]:
    row=f"N={N:<4}"
    for K in [1.5,2.0,2.5,3.0]:
        cols={p:sim(sr.DATA[p],N,K,3.0)[0] for p in CLEAN}
        port=pd.DataFrame(cols).fillna(0).mean(axis=1)
        row+=f"{sharpe(port):<8.2f}"
    print(row)

print("\nstop_atr 敏感性 (N=100,K=2.0):")
for st in [2,3,4,5]:
    cols={p:sim(sr.DATA[p],100,2.0,st)[0] for p in CLEAN}
    port=pd.DataFrame(cols).fillna(0).mean(axis=1)
    print(f"  stop={st}ATR: 夏普={sharpe(port):.2f}")

print("\n"+"="*78)
print("2) 逐品种 standalone(夏普/年交易数) + 前后半样本一致性")
print("="*78)
rows=[]
for p in CLEAN:
    d,ntr=sim(sr.DATA[p],100,2.0,3.0)
    d=d.dropna()
    mid=len(d)//2
    sh1=sharpe(d.iloc[:mid]); sh2=sharpe(d.iloc[mid:])
    yrs=(d.index[-1]-d.index[0]).days/365
    rows.append((p,sharpe(d),ntr/yrs,sh1,sh2))
R=pd.DataFrame(rows,columns=["品种","夏普","笔/年","前半夏普","后半夏普"]).sort_values("夏普",ascending=False)
print(R.round(2).to_string(index=False))
both_pos=((R.前半夏普>0)&(R.后半夏普>0)).sum()
print(f"\n前后半样本都为正的品种: {both_pos}/{len(R)}  (>0一致=非过拟合)")
print(f"前后半夏普相关: {R.前半夏普.corr(R.后半夏普):+.2f}")

print("\n"+"="*78)
print("3) 波动率目标定仓 组合资金曲线 (年化波动目标10%)")
print("="*78)
panel=pd.DataFrame(DLY).fillna(0.0).sort_index()
port_ew=panel.mean(axis=1)
# 用滚动60日组合波动缩放到10%年化
tgt=0.10/np.sqrt(252)
rollvol=port_ew.rolling(60,min_periods=20).std().shift(1)
lev=(tgt/rollvol).clip(0,3).fillna(1)
port=port_ew*lev
for nm,r in [("等权",port_ew),("波动目标10%",port)]:
    r=r.dropna();eq=(1+r).cumprod()
    days=(r.index[-1]-r.index[0]).days;ann=eq.iloc[-1]**(365/days)-1
    vol=r.std()*np.sqrt(252);sh=r.mean()/r.std()*np.sqrt(252)
    mdd=(eq/eq.cummax()-1).min();cal=ann/abs(mdd)
    segs=np.array_split(np.arange(len(r)),5)
    ss=[sharpe(r.iloc[ix]) for ix in segs]
    print(f"{nm:<14} 夏普={sh:.2f} 年化={ann:+.1%} 波动={vol:.1%} 回撤={mdd:.1%} "
          f"Calmar={cal:.2f}")
    print(f"{'':14} walk-forward 5段夏普=[{' '.join(f'{s:+.1f}' for s in ss)}] 最差={min(ss):+.1f}")

print("\n"+"="*78)
print("4) 剔除standalone亏损品种(ZN/SI)后 + 最弱段归因")
print("="*78)
CLEAN9=[p for p in CLEAN if p not in ("ZN","SI")]
cols={p:sim(sr.DATA[p],100,2.0,3.0)[0] for p in CLEAN9}
panel9=pd.DataFrame(cols).fillna(0.0).sort_index()
port9=panel9.mean(axis=1).dropna()
eq=(1+port9).cumprod();days=(port9.index[-1]-port9.index[0]).days
ann=eq.iloc[-1]**(365/days)-1;mdd=(eq/eq.cummax()-1).min()
segs=np.array_split(np.arange(len(port9)),5)
labels=[]
for ix in segs:
    sub=port9.iloc[ix]
    labels.append((sub.index[0].date(),sub.index[-1].date(),sharpe(sub)))
print(f"9品种等权(剔ZN/SI): 夏普={sharpe(port9):.2f} 年化={ann:+.1%} 回撤={mdd:.1%} Calmar={ann/abs(mdd):.2f}")
print("walk-forward 5段:")
for d0,d1,s in labels:
    print(f"  {d0}~{d1}: 夏普={s:+.2f}")

# 对照: 全市场反转(同期)在最弱两段是不是也亏 -> 确认是regime而非策略坏
print("\n最弱段(24下半年)归因: 该期商品趋势性(全品种|60日动量|均值):")
import glob
g15={f.split('/')[-1][:-8]:f for f in glob.glob('data_futures/15min/*.parquet')}
def dclose(p):
    df=pd.read_parquet(g15[p]);df['date']=pd.to_datetime(df['datetime']).dt.normalize()
    return df.groupby('date')['close'].last()
allpx=pd.DataFrame({p:dclose(p) for p in list(g15)[:40]}).sort_index()
mom=(allpx/allpx.shift(40)-1).abs().mean(axis=1)   # 截面平均|动量|=趋势强度
for d0,d1,s in labels:
    seg=mom.loc[str(d0):str(d1)].mean()
    print(f"  {d0}~{d1}: 策略夏普{s:+.2f}  市场趋势强度(|动量|均值)={seg:.3f}")
