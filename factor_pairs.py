"""假设3: 跨品种配对价差回归(产业链套利)。

经济同源对(价差应均值回归):
  RB-HC 螺纹热卷, M-RM 豆粕菜粕, Y-OI-P 油脂, L-PP 塑料, A-B 大豆,
  I-J-JM 黑色, SR vs ?, MA-V-PP 化工, BU-FU 油品, CU-ZN-AL 有色, IF-IH-IC 股指
价差 = log(P1)-beta*log(P2); 滚动z-score, 触±2反向, 回0平。日频, 成本含。
"""
import glob,numpy as np,pandas as pd
files={f.split("/")[-1][:-8]:f for f in glob.glob("data_futures/15min/*.parquet")}
def dclose(p):
    df=pd.read_parquet(files[p]); df["date"]=pd.to_datetime(df["datetime"]).dt.normalize()
    return df.groupby("date")["close"].last()
PAIRS=[("RB","HC"),("M","RM"),("Y","OI"),("Y","P"),("OI","P"),("L","PP"),("A","B"),
       ("I","J"),("J","JM"),("I","JM"),("MA","V"),("MA","PP"),("V","PP"),("BU","FU"),
       ("CU","ZN"),("CU","AL"),("ZN","AL"),("IF","IH"),("IF","IC"),("IC","IM"),
       ("SF","SM"),("AG","AU"),("SC","FU"),("PG","MA")]
def bt(p1,p2,N=60,K=2.0,cost=0.0005):
    a,b=dclose(p1),dclose(p2)
    df=pd.concat([np.log(a),np.log(b)],axis=1,keys=["a","b"]).dropna()
    if len(df)<300: return None
    beta=df["a"].rolling(N).cov(df["b"])/df["b"].rolling(N).var()
    spread=df["a"]-beta*df["b"]
    z=(spread-spread.rolling(N).mean())/spread.rolling(N).std()
    # 仓位: z>K 做空价差(空a多b), z<-K 做多价差; 回到0平
    pos=pd.Series(0.0,index=df.index); cur=0
    pv=[]
    for i in range(len(z)):
        zi=z.iloc[i]
        if cur==0:
            if zi>K: cur=-1
            elif zi<-K: cur=1
        elif cur==1 and zi>=0: cur=0
        elif cur==-1 and zi<=0: cur=0
        pv.append(cur)
    pos=pd.Series(pv,index=df.index)
    ra=a.pct_change().reindex(df.index); rb=b.pct_change().reindex(df.index)
    # 价差多头= 多a空b (beta对冲), 收益≈ ra-rb (近似等notional)
    pnl=pos.shift(1)*(ra-rb)
    turn=(pos-pos.shift(1)).abs()
    pnl=pnl-turn.shift(1).fillna(0)*cost*2
    pnl=pnl.dropna()
    if pnl.std()==0: return None
    sh=pnl.mean()/pnl.std()*np.sqrt(252)
    return sh,pnl,(pos!=0).mean()
print("配对价差回归 (滚动60日z, 触±2反向, 净5bp/边×2腿):")
res={}
allpnl=[]
for p1,p2 in PAIRS:
    if p1 not in files or p2 not in files: continue
    r=bt(p1,p2)
    if r is None: continue
    sh,pnl,expo=r; res[f"{p1}-{p2}"]=pnl
    allpnl.append(pnl)
    print(f"  {p1}-{p2:<3} 夏普={sh:>5.2f} 持仓占比={expo:.0%}")
# 等权组合所有配对
port=pd.concat(allpnl,axis=1).fillna(0).mean(axis=1)
sh=port.mean()/port.std()*np.sqrt(252); days=(port.index[-1]-port.index[0]).days
ann=(1+port).prod()**(365/days)-1
segs=np.array_split(np.arange(len(port)),5)
ss=[port.iloc[ix].mean()/port.iloc[ix].std()*np.sqrt(252) if port.iloc[ix].std()>0 else 0 for ix in segs]
print(f"\n等权组合所有配对: 夏普={sh:.2f} 年化={ann:+.1%} 5段=[{' '.join(f'{s:+.1f}' for s in ss)}] 最差={min(ss):+.1f}")
