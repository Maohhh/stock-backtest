import numpy as np, pandas as pd
import strategy_range as sr

def sim_count(df, N, K, stop_atr, cost):
    C=df["close"];H=df["high"];L=df["low"]
    mid=C.rolling(N,min_periods=N).mean(); sd=C.rolling(N,min_periods=N).std()
    up=(mid+K*sd).to_numpy(); lo=(mid-K*sd).to_numpy(); midn=mid.to_numpy()
    c=C.to_numpy();h=H.to_numpy();l=L.to_numpy()
    tr=pd.concat([H-L,(H-C.shift()).abs(),(L-C.shift()).abs()],axis=1).max(axis=1)
    atr=tr.rolling(sr.ATR_N,min_periods=sr.ATR_N).mean().to_numpy()
    n=len(c);warm=max(N,sr.ATR_N)+1;rb=np.zeros(n);p=0;ei=-1;estop=0.0
    ntr=0; holds=[]
    for i in range(warm,n):
        if np.isnan(midn[i]) or np.isnan(atr[i-1]): continue
        ret=0.0
        if p!=0:
            px=c[i];ex=False
            if p>0:
                if l[i]<=estop: px=estop;ex=True
                elif c[i]>=midn[i] or (i-ei)>=sr.MAXH: ex=True
            else:
                if h[i]>=estop: px=estop;ex=True
                elif c[i]<=midn[i] or (i-ei)>=sr.MAXH: ex=True
            ret=p*(px/c[i-1]-1)
            if ex: ret-=cost;p=0;ntr+=1;holds.append(i-ei)
        rb[i]=ret
        if p==0:
            if c[i]<lo[i]: p=1;ei=i;estop=c[i]-stop_atr*atr[i];rb[i]-=cost
            elif c[i]>up[i]: p=-1;ei=i;estop=c[i]+stop_atr*atr[i];rb[i]-=cost
    s=pd.Series(rb,index=pd.to_datetime(df["datetime"]))
    return s.groupby(s.index.normalize()).sum(), ntr, (np.mean(holds) if holds else 0)

groups=sr.make_groups()
for gname in ["REVERT(最回归1/3)","CSTYPE(低波回归)"]:
    prods=groups[gname]
    print(f"\n===== {gname}  ({len(prods)}品种) =====")
    # 单品种standalone夏普
    solos=[]
    for p in prods:
        d,_,_=sim_count(sr.DATA[p],100,2.0,3.0,0.0002)
        d=d.dropna()
        if d.std()>0: solos.append(d.mean()/d.std()*np.sqrt(252))
    print(f"单品种standalone夏普: 中位={np.median(solos):.2f} 均值={np.mean(solos):.2f} "
          f"范围[{min(solos):.1f},{max(solos):.1f}]")
    # 成本敏感性
    print("成本敏感性(组合夏普):")
    for cost in [0.0002,0.0005,0.001,0.002]:
        cols={};tot_tr=0;hh=[]
        for p in prods:
            d,nt,hb=sim_count(sr.DATA[p],100,2.0,3.0,cost)
            cols[p]=d;tot_tr+=nt;hh.append(hb)
        port=pd.DataFrame(cols).fillna(0).mean(axis=1).dropna()
        sh=port.mean()/port.std()*np.sqrt(252)
        days=(port.index[-1]-port.index[0]).days
        ann=(1+port).prod()**(365/days)-1
        print(f"  成本{cost*1e4:>4.0f}bp/边: 夏普={sh:>5.2f} 年化={ann:>+6.1%} "
              f"总交易={tot_tr} (~{tot_tr/len(prods)/(days/365):.0f}笔/品种/年, 均持{np.mean(hh):.0f}根)")
