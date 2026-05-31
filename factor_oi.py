"""假设2: 持仓量(OI)结构因子。数据已含 open_interest。

测两类信号(日频, 全品种panel, 截面多空中性 + 时序):
  A) OI动量: ΔOI 与未来收益(增仓方向延续?)
  B) 价-仓配合: 价涨+增仓=趋势确认; 价涨+减仓=空头回补将衰竭
  C) 截面: 按ΔOI%排序多空
低换手优先(日频, 不做日内)。
"""
import glob, numpy as np, pandas as pd
files=sorted(glob.glob("data_futures/15min/*.parquet"))

def daily(df):
    df=df.copy(); df["date"]=pd.to_datetime(df["datetime"]).dt.normalize()
    g=df.groupby("date").agg(close=("close","last"),oi=("open_interest","last"),
                             vol=("volume","sum"))
    return g

D={}
for f in files:
    p=f.split("/")[-1][:-8]
    d=daily(pd.read_parquet(f))
    if len(d)>300: D[p]=d
close=pd.DataFrame({p:D[p]["close"] for p in D}).sort_index()
oi=pd.DataFrame({p:D[p]["oi"] for p in D}).reindex(close.index)
ret=close.pct_change()
doi=oi.pct_change()                       # ΔOI%
print(f"panel: {close.shape[0]}日 × {close.shape[1]}品种")

def perf(pnl,label,cost=0.0,turn=None):
    pnl=pnl.dropna()
    if turn is not None and cost>0: pnl=pnl-turn.reindex(pnl.index).shift(1).fillna(0)*cost
    if pnl.std()==0 or len(pnl)<50: print(f"{label}: 无效"); return
    sh=pnl.mean()/pnl.std()*np.sqrt(252)
    days=(pnl.index[-1]-pnl.index[0]).days; ann=(1+pnl).prod()**(365/days)-1
    segs=np.array_split(np.arange(len(pnl)),5)
    ss=[pnl.iloc[ix].mean()/pnl.iloc[ix].std()*np.sqrt(252) if pnl.iloc[ix].std()>0 else 0 for ix in segs]
    print(f"{label:<34} 夏普={sh:>5.2f} 年化={ann:>+6.1%} 5段=[{' '.join(f'{s:+.1f}' for s in ss)}] 最差={min(ss):+.1f}")

# A) 截面 ΔOI 动量/反转
print("\n--- A) 截面按 ΔOI% 排序 (5日均ΔOI, 多空中性, 净5bp) ---")
for L in [1,3,5,10]:
    sig=doi.rolling(L).mean()
    x=sig.sub(sig.mean(axis=1),axis=0); w=x.div(x.abs().sum(axis=1),axis=0)  # 增仓多
    for sign,nm in [(1,"增仓做多"),(-1,"增仓做空")]:
        ww=sign*w; pnl=(ww.shift(1)*ret).sum(axis=1); turn=(ww-ww.shift(1)).abs().sum(axis=1)
        perf(pnl,f"  L={L} {nm}",0.0005,turn)

# B) 价-仓配合 时序: 各品种 sign(ret)*sign(doi), 截面中性化后看
print("\n--- B) 截面 价×仓 配合 (price-OI 共振, 净5bp) ---")
for L in [3,5,10]:
    pm=close/close.shift(L)-1; om=doi.rolling(L).mean()
    confirm=np.sign(pm)*np.sign(om)            # +1共振(增仓助涨/助跌)
    sig=pm*confirm                              # 趋势确认强度
    x=sig.sub(sig.mean(axis=1),axis=0); w=x.div(x.abs().sum(axis=1),axis=0)
    pnl=(w.shift(1)*ret).sum(axis=1);turn=(w-w.shift(1)).abs().sum(axis=1)
    perf(pnl,f"  L={L} 价仓共振动量",0.0005,turn)
    perf(-pnl,f"  L={L} 价仓共振反转",0.0005,turn)
