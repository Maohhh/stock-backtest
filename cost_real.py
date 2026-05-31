"""用真实tick重算成本: 干净可交易集在现实成本下到底能不能成立。

成本结构(期货真实): 单边成本 = 滑点 + 手续费
  滑点: 主流品种买卖价差≈1个tick; 来回穿价差≈1 tick = tick_size/price (占比)
        -> 单边≈半个tick; 激进单(市价)最多1 tick/边
  手续费: 主流品种约 0.5~1.5bp/边
真实单边 ≈ 0.5*tick/price + 1bp(手续) ; 保守(市价双向) ≈ 1*tick/price + 1bp
"""
import glob, numpy as np, pandas as pd
import strategy_range as sr

# 干净集 tick_size (元/吨 或 点)
TICK = {"V":1,"PX":2,"MA":1,"SI":5,"ZN":5,"SH":1,"BU":1,"RM":1,"UR":1,"PG":1,"AP":1}
CLEAN = list(TICK)

print("各品种真实tick占价格比例(=1个tick的相对成本):")
tickpct = {}
for p in CLEAN:
    px = sr.DATA[p]["close"].median()
    tp = TICK[p] / px
    tickpct[p] = tp
    print(f"  {p:4s} tick={TICK[p]} 中位价={px:>8.0f} -> 1tick={tp*1e4:>4.1f}bp")
avg = np.mean(list(tickpct.values()))
print(f"平均 1tick = {avg*1e4:.1f}bp")
print(f"=> 现实单边成本估计: 乐观(限价)≈{0.5*avg*1e4+0.5:.1f}bp, "
      f"中性≈{0.5*avg*1e4+1:.1f}bp, 保守(市价)≈{avg*1e4+1.5:.1f}bp")

def wf(prods, cost_map):
    cols={}
    for p in prods:
        c = cost_map[p] if isinstance(cost_map,dict) else cost_map
        cols[p]=sr.sim_daily(sr.DATA[p],100,2.0,3.0)  # 先拿无成本... 需改
    return None

# strategy_range.sim_daily 用固定 sr.COST. 重新实现带 per-product cost
def sim_daily_cost(df, N, K, stop_atr, cost):
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

def run(cost_desc, costfn):
    cols={p:sim_daily_cost(sr.DATA[p],100,2.0,3.0,costfn(p)) for p in CLEAN}
    port=pd.DataFrame(cols).fillna(0).mean(axis=1).sort_index().dropna()
    sh=port.mean()/port.std()*np.sqrt(252);days=(port.index[-1]-port.index[0]).days
    ann=(1+port).prod()**(365/days)-1
    mdd=((1+port).cumprod()/(1+port).cumprod().cummax()-1).min()
    segs=np.array_split(np.arange(len(port)),5)
    ss=[port.iloc[ix].mean()/port.iloc[ix].std()*np.sqrt(252) if port.iloc[ix].std()>0 else 0 for ix in segs]
    print(f"{cost_desc:<28} 夏普={sh:>5.2f} 年化={ann:>+6.1%} 回撤={mdd:>6.1%} 5段=[{' '.join(f'{s:+.1f}' for s in ss)}] 最差={min(ss):+.1f}")

print("\n干净集(11品种)区间反转 在不同成本下:")
for bp in [1,2,3,5,10]:
    run(f"  固定{bp}bp/边", lambda p,bp=bp: bp/1e4)
print("  --- 按各品种真实tick ---")
run("  乐观(0.5tick+0.5bp手续)", lambda p: 0.5*tickpct[p]+0.00005)
run("  中性(0.75tick+1bp手续)",  lambda p: 0.75*tickpct[p]+0.0001)
run("  保守(1tick+1.5bp手续)",   lambda p: 1.0*tickpct[p]+0.00015)
