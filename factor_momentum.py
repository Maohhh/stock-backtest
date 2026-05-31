"""假设4: 动量(商品期货最经典稳健的因子)。

时序动量(TSMOM, Moskowitz): 各品种按过去L日收益符号做多/空。
截面动量(XSMOM, Asness): 跨品种按过去L日收益排序, 多强空弱。
对比反转(我们之前测的)。日频, 多空中性(截面)/等权(时序), 含成本。
"""
import glob,numpy as np,pandas as pd
files=sorted(glob.glob("data_futures/15min/*.parquet"))
def dclose(f):
    df=pd.read_parquet(f); df["date"]=pd.to_datetime(df["datetime"]).dt.normalize()
    return df.groupby("date")["close"].last()
close=pd.DataFrame({f.split("/")[-1][:-8]:dclose(f) for f in files}).sort_index()
close=close.dropna(thresh=int(len(close.columns)*0.6))
ret=close.pct_change()
vol=ret.rolling(40).std()                     # 用于波动率归一(风险平价)
print(f"panel: {close.shape[0]}日 × {close.shape[1]}品种")

def perf(pnl,turn,label,cost=0.0005):
    pnl=(pnl-turn.shift(1).fillna(0)*cost).dropna()
    if pnl.std()==0 or len(pnl)<50: print(f"{label}: 无效");return
    sh=pnl.mean()/pnl.std()*np.sqrt(252); days=(pnl.index[-1]-pnl.index[0]).days
    ann=(1+pnl).prod()**(365/days)-1
    segs=np.array_split(np.arange(len(pnl)),5)
    ss=[pnl.iloc[ix].mean()/pnl.iloc[ix].std()*np.sqrt(252) if pnl.iloc[ix].std()>0 else 0 for ix in segs]
    print(f"{label:<32} 夏普={sh:>5.2f} 年化={ann:>+6.1%} 5段=[{' '.join(f'{s:+.1f}' for s in ss)}] 最差={min(ss):+.1f}")

print("\n--- 时序动量 TSMOM (做多正动量/做空负动量, 波动率归一, 等权) ---")
for L in [20,40,60,120]:
    mom=close/close.shift(L)-1
    w=np.sign(mom)/vol                         # 风险平价
    w=w.div(w.abs().sum(axis=1),axis=0)
    pnl=(w.shift(1)*ret).sum(axis=1); turn=(w-w.shift(1)).abs().sum(axis=1)
    perf(pnl,turn,f"  TSMOM L={L}")

print("\n--- 截面动量 XSMOM (多强空弱, 多空中性) ---")
for L in [20,40,60,120]:
    mom=close/close.shift(L)-1
    x=mom.sub(mom.mean(axis=1),axis=0); w=x.div(x.abs().sum(axis=1),axis=0)
    pnl=(w.shift(1)*ret).sum(axis=1); turn=(w-w.shift(1)).abs().sum(axis=1)
    perf(pnl,turn,f"  XSMOM L={L} (动量)")
    perf(-pnl,turn,f"  XSMOM L={L} (反转,对照)")
