"""第三类交易方法: 波动率交易 + 季节性(用上'农业周期'洞察)。

A) 波动率: 不赌方向, 赌波动状态。
   A1 波动突破: 低波后的突破延续(波动聚集) — 时序
   A2 波动率均值回归本身: 高ATR->做空波动(已含在反转里, 这里测"波动放大预示大波")
B) 季节性: 农产品月份效应 — 同一日历月的历史平均收益(留一法防前视)。
口径: 日频, 含真实成本2.5bp, walk-forward。
"""
import glob, numpy as np, pandas as pd
files=sorted(glob.glob("data_futures/15min/*.parquet"))
COST=0.00025
def dbars(f):
    df=pd.read_parquet(f);df["date"]=pd.to_datetime(df["datetime"]).dt.normalize()
    return df.groupby("date").agg(close=("close","last"),high=("high","max"),
                                  low=("low","min"))
D={f.split("/")[-1][:-8]:dbars(f) for f in files}
D={p:d for p,d in D.items() if len(d)>400}
close=pd.DataFrame({p:D[p]["close"] for p in D}).sort_index()
ret=close.pct_change()
print(f"panel {close.shape[0]}日 × {close.shape[1]}品种")

def perf(pnl,turn,label,cost=COST):
    pnl=(pnl-turn.shift(1).fillna(0)*cost).dropna()
    if pnl.std()==0 or len(pnl)<50:print(f"{label}:无效");return
    sh=pnl.mean()/pnl.std()*np.sqrt(252);days=(pnl.index[-1]-pnl.index[0]).days
    ann=(1+pnl).prod()**(365/days)-1
    segs=np.array_split(np.arange(len(pnl)),5)
    ss=[pnl.iloc[ix].mean()/pnl.iloc[ix].std()*np.sqrt(252) if pnl.iloc[ix].std()>0 else 0 for ix in segs]
    print(f"{label:<30} 夏普={sh:>5.2f} 年化={ann:>+6.1%} 5段=[{' '.join(f'{s:+.1f}' for s in ss)}] 最差={min(ss):+.1f}")

# ATR / 波动
tr=pd.DataFrame({p:np.maximum(D[p]["high"]-D[p]["low"],
        np.maximum((D[p]["high"]-D[p]["close"].shift()).abs(),
                   (D[p]["low"]-D[p]["close"].shift()).abs())) for p in D})
atr=tr.rolling(20).mean()

print("\n=== A) 波动率交易 ===")
# A1 波动突破(Donchian风格): 收盘突破N日高/低 且 波动放大 -> 顺势(动量在低波后)
for N in [20,40]:
    hh=close.rolling(N).max().shift(1);ll=close.rolling(N).min().shift(1)
    brk=np.where(close>hh,1,np.where(close<ll,-1,np.nan))
    w=pd.DataFrame(brk,index=close.index,columns=close.columns).ffill(limit=N)
    w=w.div(atr).div(close)        # 波动率归一名义
    w=w.div(w.abs().sum(axis=1),axis=0)
    pnl=(w.shift(1)*ret).sum(axis=1);turn=(w-w.shift(1)).abs().sum(axis=1)
    perf(pnl,turn,f"  A1 {N}日通道突破(顺势)")

# A2 波动状态过滤: 仅在低波环境做均值回归 vs 高波环境
vol_state=atr.div(close).rank(axis=0,pct=True)  # 时序分位(高=当前高波)
# 简化: 整体方向中性, 看"波动扩张"日后的绝对收益(波动聚集=明日波动大)
dvol=(atr.div(close)).pct_change()
print("  A2 波动聚集验证: 今日ATR扩张 -> 明日|收益|相关")
ac=[]
for p in D:
    x=(atr[p]/close[p]).pct_change(); y=ret[p].abs().shift(-1)
    ac.append(x.corr(y))
print(f"     ATR变化 vs 次日|收益| 相关均值={np.nanmean(ac):+.3f} (>0=波动可预测,可做跨式)")

print("\n=== B) 季节性(农业周期) ===")
# 月份效应: 每个品种每个日历月的平均日收益, 留一年防前视
mret=ret.copy()
mret["ym"]=mret.index.to_period("M");mret["m"]=mret.index.month;mret["y"]=mret.index.year
sig=pd.DataFrame(0.0,index=ret.index,columns=close.columns)
for p in close.columns:
    s=ret[p]
    for dt in ret.index:
        # 用"除当前年外"该月历史均值作为信号(防前视)
        hist=s[(s.index.month==dt.month)&(s.index.year<dt.year)]
        if len(hist)>15: sig.loc[dt,p]=np.sign(hist.mean())
# 截面季节性: 多预期涨/空预期跌, 月度持有
w=sig.div(sig.abs().sum(axis=1).replace(0,1),axis=0)
reb=ret.index[::20]
mask=pd.Series(ret.index.isin(reb),index=ret.index)
w=w.where(mask,np.nan).ffill()
pnl=(w.shift(1)*ret).sum(axis=1);turn=(w-w.shift(1)).abs().sum(axis=1)
perf(pnl,turn,"  B 季节性月份效应(全品种)")
# 只在农产品上
agri=[p for p in ["A","B","M","Y","P","C","CS","RM","OI","SR","CF","AP","JD","CJ","PK","LH","UR"] if p in close.columns]
wa=sig[agri].div(sig[agri].abs().sum(axis=1).replace(0,1),axis=0)
wa=wa.where(mask,np.nan).ffill()
pnla=(wa.shift(1)*ret[agri]).sum(axis=1);turna=(wa-wa.shift(1)).abs().sum(axis=1)
perf(pnla,turna,f"  B 季节性(仅{len(agri)}个农产品)")
