"""假设1: carry/期限结构因子 (期货独有, 低换手, 不拥挤)。

数据: data_futures/carry/{prod}.parquet, 日频 carry=年化展期收益(近月-远月)。
因子逻辑: backwardation(carry>0,现货溢价)的品种长期跑赢contango(carry<0)。
  截面: 按carry排序, 多空中性(多高carry空低carry)。
  时序: carry>0做多/carry<0做空各品种。
按月调仓(carry是慢变量, 天然低换手)。
"""
import glob,numpy as np,pandas as pd
cf=glob.glob("data_futures/carry/*.parquet")
if len(cf)<10:
    print(f"carry数据不足({len(cf)}), 等下载完成"); exit()
carry={}; nearp={}
for f in cf:
    p=f.split("/")[-1][:-8]
    d=pd.read_parquet(f).set_index("date")
    carry[p]=d["carry"]; nearp[p]=d["near"]
C=pd.DataFrame(carry).sort_index()
P=pd.DataFrame(nearp).sort_index()           # 近月价格(算收益)
# 用15min加权价做收益更连续
g15={f.split("/")[-1][:-8]:f for f in glob.glob("data_futures/15min/*.parquet")}
def dclose(p):
    df=pd.read_parquet(g15[p]);df["date"]=pd.to_datetime(df["datetime"]).dt.normalize()
    return df.groupby("date")["close"].last()
common=[p for p in C.columns if p in g15]
px=pd.DataFrame({p:dclose(p) for p in common}).reindex(C.index).sort_index()
C=C[common]
ret=px.pct_change()
# carry平滑(20日)去噪, winsorize
Cs=C.rolling(20,min_periods=5).mean()
print(f"carry panel: {C.shape[0]}日 × {len(common)}品种")
print(f"carry截面分布: 均值{C.stack().mean():+.3f} 标准差{C.stack().std():.3f}")

def perf(pnl,turn,label,cost=0.0005):
    pnl=(pnl-turn.shift(1).fillna(0)*cost).dropna()
    if pnl.std()==0 or len(pnl)<50:print(f"{label}: 无效");return
    sh=pnl.mean()/pnl.std()*np.sqrt(252);days=(pnl.index[-1]-pnl.index[0]).days
    ann=(1+pnl).prod()**(365/days)-1
    segs=np.array_split(np.arange(len(pnl)),5)
    ss=[pnl.iloc[ix].mean()/pnl.iloc[ix].std()*np.sqrt(252) if pnl.iloc[ix].std()>0 else 0 for ix in segs]
    print(f"{label:<30} 夏普={sh:>5.2f} 年化={ann:>+6.1%} 5段=[{' '.join(f'{s:+.1f}' for s in ss)}] 最差={min(ss):+.1f}")

print("\n--- 截面 carry (多高carry/空低carry, 多空中性, 月度调仓) ---")
reb=C.index[::20]                            # 每20交易日调仓
for q in [0.33]:
    x=Cs.sub(Cs.median(axis=1),axis=0)
    w=x.div(x.abs().sum(axis=1),axis=0)
    w=w.reindex(C.index).where(C.index.isin(reb)).ffill()  # 月度持有
    pnl=(w.shift(1)*ret).sum(axis=1);turn=(w-w.shift(1)).abs().sum(axis=1)
    perf(pnl,turn,"  截面carry(月调)")
    perf(-pnl,turn,"  截面carry反向(对照)")

print("\n--- 时序 carry (carry符号定多空, 月度) ---")
w=np.sign(Cs); w=w.div(w.abs().sum(axis=1),axis=0)
w=w.reindex(C.index).where(C.index.isin(reb)).ffill()
pnl=(w.shift(1)*ret).sum(axis=1);turn=(w-w.shift(1)).abs().sum(axis=1)
perf(pnl,turn,"  时序carry(月调)")

print("\n--- 截面 carry 日度调仓(对照换手) ---")
x=Cs.sub(Cs.median(axis=1),axis=0);w=x.div(x.abs().sum(axis=1),axis=0)
pnl=(w.shift(1)*ret).sum(axis=1);turn=(w-w.shift(1)).abs().sum(axis=1)
perf(pnl,turn,"  截面carry(日调)")
