import glob,numpy as np,pandas as pd
cf=glob.glob("data_futures/carry/*.parquet")
carry={};
for f in cf:
    p=f.split("/")[-1][:-8];d=pd.read_parquet(f).set_index("date")
    if len(d)>=200: carry[p]=d["carry"]   # 剔除国债等短序列
C=pd.DataFrame(carry).sort_index()
g15={f.split("/")[-1][:-8]:f for f in glob.glob("data_futures/15min/*.parquet")}
def dclose(p):
    df=pd.read_parquet(g15[p]);df["date"]=pd.to_datetime(df["datetime"]).dt.normalize()
    return df.groupby("date")["close"].last()
common=[p for p in C.columns if p in g15]; C=C[common]
px=pd.DataFrame({p:dclose(p) for p in common}).reindex(C.index).sort_index()
ret=px.pct_change()
Cs=C.rolling(20,min_periods=5).mean()
print(f"剔短序列后 {len(common)}品种")

# 分5层(按carry), 看各层未来20日收益 单调性
fwd=px.shift(-20)/px-1
rows=[]
for i in range(40,len(C)-20,5):
    row=Cs.iloc[i].dropna()
    if len(row)<15: continue
    qs=pd.qcut(row,5,labels=False,duplicates='drop')
    fr=fwd.iloc[i]
    for q in range(5):
        members=qs[qs==q].index
        rows.append((q,fr[members].mean()))
df=pd.DataFrame(rows,columns=['q','fwd']).dropna()
print("\ncarry五分位 未来20日平均收益(Q0=最contango..Q4=最backwardation):")
print(df.groupby('q').fwd.mean().round(4).to_string())
print("\n若学术carry成立应单调递增(Q4>Q0); 实际:")
m=df.groupby('q').fwd.mean()
print(f"Q4-Q0 = {m.iloc[-1]-m.iloc[0]:+.4f}  ({'backwardation跑赢(学术)' if m.iloc[-1]>m.iloc[0] else 'contango跑赢(反学术)'})")

# 反向carry收益是否被少数品种主导: 看每品种 sign(-carry)*fwd 的贡献
print("\n反向carry(多contango)各品种平均日贡献 top/bottom:")
sig=-np.sign(Cs)
contrib=(sig.shift(1)*ret).mean().sort_values()
print("最负(拖累):",contrib.head(4).round(5).to_dict())
print("最正(主导):",contrib.tail(4).round(5).to_dict())
