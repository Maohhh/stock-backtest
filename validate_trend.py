"""趋势突破做实: 参数稳健性 + 逐品种standalone + 前后半一致性(真实成本2.5bp)。
对标反转的validate_proto, 同样严格。"""
import glob, numpy as np, pandas as pd
COST=0.00025
files=sorted(glob.glob("data_futures/15min/*.parquet"))
def dbars(f):
    df=pd.read_parquet(f);df["date"]=pd.to_datetime(df["datetime"]).dt.normalize()
    return df.groupby("date").agg(close=("close","last"),high=("high","max"),low=("low","min"))
D={f.split("/")[-1][:-8]:dbars(f) for f in files}
D={p:d for p,d in D.items() if len(d)>400}
close=pd.DataFrame({p:D[p]["close"] for p in D}).sort_index();ret=close.pct_change()
tr=pd.DataFrame({p:np.maximum(D[p]["high"]-D[p]["low"],np.maximum(
    (D[p]["high"]-D[p]["close"].shift()).abs(),(D[p]["low"]-D[p]["close"].shift()).abs())) for p in D})
atr=tr.rolling(20).mean()

def trend_w(N):
    hh=close.rolling(N).max().shift(1);ll=close.rolling(N).min().shift(1)
    brk=np.where(close>hh,1,np.where(close<ll,-1,np.nan))
    w=pd.DataFrame(brk,index=close.index,columns=close.columns).ffill(limit=N)
    return (w/atr/close)            # 波动率归一(未截面归一, 单品种用)

def sh(s):s=s.dropna();return s.mean()/s.std()*np.sqrt(252) if s.std()>0 else 0

print("="*72);print("1) 参数稳健性: 通道N (全品种组合夏普, 真实成本)");print("="*72)
for N in [10,20,30,40,55,80,120]:
    w=trend_w(N);wn=w.div(w.abs().sum(axis=1),axis=0)
    pnl=(wn.shift(1)*ret).sum(axis=1);turn=(wn-wn.shift(1)).abs().sum(axis=1)
    p=(pnl-turn.shift(1).fillna(0)*COST)
    print(f"  N={N:<3}: 夏普={sh(p):.2f}")

print("\n"+"="*72);print("2) 逐品种standalone(N=40) + 前后半一致性");print("="*72)
N=40;w=trend_w(N)
rows=[]
for p in D:
    wp=w[p]/w[p].abs().rolling(60,min_periods=10).mean()   # 自身归一到~1单位
    wp=np.sign(w[p])  # 趋势方向(±1), 单品种全仓方向
    pnl=(wp.shift(1)*ret[p]); turn=(wp-wp.shift(1)).abs()
    pl=(pnl-turn.shift(1).fillna(0)*COST).dropna()
    if len(pl)<200: continue
    mid=len(pl)//2
    rows.append((p,sh(pl),sh(pl.iloc[:mid]),sh(pl.iloc[mid:])))
R=pd.DataFrame(rows,columns=["品种","夏普","前半","后半"]).sort_values("夏普",ascending=False)
print("Top10:");print(R.head(10).round(2).to_string(index=False))
print("Bottom5:");print(R.tail(5).round(2).to_string(index=False))
bp=((R.前半>0)&(R.后半>0)).sum()
print(f"\n全品种({len(R)}个)中位夏普={R.夏普.median():.2f}  前后半都正={bp}/{len(R)} "
      f"前后半相关={R.前半.corr(R.后半):+.2f}")
print(f"正夏普品种数={(R.夏普>0).sum()}/{len(R)}")
