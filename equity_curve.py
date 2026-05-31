import numpy as np, pandas as pd
import strategy_range as sr

COST=0.00025
CLEAN9=["V","PX","MA","SH","BU","RM","UR","PG","AP"]

def sim(df,N,K,stop_atr,cost=COST):
    C=df["close"];H=df["high"];L=df["low"]
    mid=C.rolling(N,min_periods=N).mean();sd=C.rolling(N,min_periods=N).std()
    up=(mid+K*sd).to_numpy();lo=(mid-K*sd).to_numpy();midn=mid.to_numpy()
    c=C.to_numpy();h=H.to_numpy();l=L.to_numpy()
    tr=pd.concat([H-L,(H-C.shift()).abs(),(L-C.shift()).abs()],axis=1).max(axis=1)
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

panel=pd.DataFrame({p:sim(sr.DATA[p],100,2.0,3.0) for p in CLEAN9}).fillna(0).sort_index()
# 等权组合日收益(每个品种1单位名义, 组合=均值 -> 相当于总名义=本金, 各品种1/9)
port=panel.mean(axis=1).dropna()
days=(port.index[-1]-port.index[0]).days; yrs=days/365
eq=(1+port).cumprod()
total=eq.iloc[-1]; ann=total**(365/days)-1
mdd=(eq/eq.cummax()-1).min()
vol=port.std()*np.sqrt(252)
print(f"回测期: {port.index[0].date()} ~ {port.index[-1].date()}  ({yrs:.2f}年, {len(port)}交易日)")
print(f"组合(9品种等权, 名义敞口≈1倍本金) 真实成本2.5bp:")
print(f"  累计倍数={total:.3f}  年化={ann:+.1%}  年化波动={vol:.1%}  最大回撤={mdd:.1%}")
print()
cap0=100000
print(f"=== 本金10万, 名义敞口=1倍本金(不加杠杆) ===")
print(f"  {yrs:.1f}年后 = {cap0*total:,.0f} 元  (年化{ann:.1%})")
for y in [1,2,3,5,10]:
    print(f"  按年化{ann:.1%}复利 {y:>2}年: {cap0*(1+ann)**y:,.0f} 元")

print(f"\n=== 期货保证金特性: 1倍名义敞口实际只占用~10-15%保证金 ===")
print(f"若把名义敞口放大到2倍本金(仍很保守, 占用~25%保证金):")
port2=port*2; eq2=(1+port2).cumprod(); ann2=eq2.iloc[-1]**(365/days)-1; mdd2=(eq2/eq2.cummax()-1).min()
print(f"  年化={ann2:+.1%} 最大回撤={mdd2:.1%}  10万 -> {yrs:.1f}年后 {cap0*eq2.iloc[-1]:,.0f}元")
print(f"放大到3倍名义(占用~40%保证金, 接近你说的'全仓'):")
port3=port*3; eq3=(1+port3).cumprod(); ann3=eq3.iloc[-1]**(365/days)-1; mdd3=(eq3/eq3.cummax()-1).min()
print(f"  年化={ann3:+.1%} 最大回撤={mdd3:.1%}  10万 -> {yrs:.1f}年后 {cap0*eq3.iloc[-1]:,.0f}元")

# 单品种"全仓"对照(真正每次全仓单品种=最高风险)
print(f"\n=== 若真'每次全仓单一品种'(最危险) 各品种单独复利 ===")
for p in CLEAN9:
    e=(1+sim(sr.DATA[p],100,2.0,3.0).dropna()).cumprod()
    a=e.iloc[-1]**(365/days)-1; m=(e/e.cummax()-1).min()
    print(f"  {p}: 年化{a:+6.1%} 回撤{m:6.1%} 10万->{cap0*e.iloc[-1]:>10,.0f}元")
