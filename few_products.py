"""15min级别, 3-5个品种, 找最赚的配置。
①反转在最强单品种/3品种/5品种的表现 ②叠加第二个15min独立edge(日内时段/突破)"""
import glob, numpy as np, pandas as pd
import strategy_range as sr, equity_curve as ec
COST=0.00025
def sh(s):s=s.dropna();return s.mean()/s.std()*np.sqrt(252) if s.std()>0 else 0
def rep(s,nm):
    s=s.dropna();eq=(1+s).cumprod();d=(s.index[-1]-s.index[0]).days;ann=eq.iloc[-1]**(365/d)-1
    mdd=(eq/eq.cummax()-1).min();seg=np.array_split(np.arange(len(s)),5);ss=[sh(s.iloc[ix]) for ix in seg]
    print(f"{nm:<26} 夏普={sh(s):>5.2f} 年化={ann:>+6.1%} 回撤={mdd:>6.1%} 5段=[{' '.join(f'{x:+.1f}' for x in ss)}]")

# 反转: 各品种单独
CLEAN=["V","PX","MA","SH","BU","RM","UR","PG","AP"]
solo={p:ec.sim(sr.DATA[p],100,2.0,3.0) for p in CLEAN}
ranked=sorted(CLEAN,key=lambda p:sh(solo[p]),reverse=True)
print("反转·各品种standalone夏普(降序):")
for p in ranked: print(f"  {p}: {sh(solo[p]):.2f}")
print()
panel=pd.DataFrame(solo).fillna(0)
rep(panel[ranked[:1]].mean(axis=1),"反转·最强1品种("+ranked[0]+")")
rep(panel[ranked[:3]].mean(axis=1),"反转·最强3品种("+",".join(ranked[:3])+")")
rep(panel[ranked[:5]].mean(axis=1),"反转·最强5品种")
rep(panel[ranked].mean(axis=1),"反转·全9品种(对照)")

print("\n=== 关键检验: 用前半段选品种, 后半段实测(防前视) ===")
# 全部可交易回归型品种(不止9个, 给选择更大池子)
POOL=["V","PX","MA","SH","BU","RM","UR","PG","AP","SI","ZN","SR","SP","L","PP","CF","SS","HC","FG"]
POOL=[p for p in POOL if p in sr.DATA]
allsolo={p:ec.sim(sr.DATA[p],100,2.0,3.0) for p in POOL}
allp=pd.DataFrame(allsolo).fillna(0).sort_index()
mid=len(allp)//2
train=allp.iloc[:mid]; test=allp.iloc[mid:]
def shc(s):return s.mean()/s.std()*np.sqrt(252) if s.std()>0 else 0
train_sh={p:shc(train[p]) for p in POOL}
for k in [3,5,8]:
    top=sorted(POOL,key=lambda p:train_sh[p],reverse=True)[:k]
    test_port=test[top].mean(axis=1)
    full_port=allp[top].mean(axis=1)
    print(f"前半选最强{k} {top}")
    print(f"  -> 后半(样本外)夏普={shc(test_port):.2f}  全程夏普={shc(full_port):.2f}")
# 对照: 固定全池等权(不选)
print(f"\n对照·全池{len(POOL)}品种等权 后半夏普={shc(test[POOL].mean(axis=1)):.2f}")
print(f"对照·随机5品种(按字母)后半夏普={shc(test[POOL[:5]].mean(axis=1)):.2f}")

print("\n=== 实操版: 滚动选品种(每季度按过去半年夏普选最强5, 全程无前视) ===")
dates=allp.index
def shc2(s):return s.mean()/s.std()*np.sqrt(252) if (len(s)>20 and s.std()>0) else -9
port=pd.Series(0.0,index=dates)
lookback=120; rebal=60; K=5
held=[]
picks_log=[]
for i in range(lookback,len(dates)):
    if (i-lookback)%rebal==0:  # 每60交易日重选
        win=allp.iloc[i-lookback:i]
        held=sorted(POOL,key=lambda p:shc2(win[p]),reverse=True)[:K]
        picks_log.append((dates[i].date(),held.copy()))
    if held:
        port.iloc[i]=allp[held].iloc[i].mean()
port=port.iloc[lookback:]
eq=(1+port).cumprod();d=(port.index[-1]-port.index[0]).days;ann=eq.iloc[-1]**(365/d)-1
mdd=(eq/eq.cummax()-1).min();seg=np.array_split(np.arange(len(port)),5);ss=[shc2(port.iloc[ix]) for ix in seg]
print(f"滚动选5品种: 夏普={shc2(port):.2f} 年化={ann:+.1%} 回撤={mdd:.1%} 5段=[{' '.join(f'{x:+.1f}' for x in ss)}]")
print(f"10万 -> {100000*eq.iloc[-1]:,.0f}元")
print("近几次选中的品种:")
for dt,pk in picks_log[-4:]: print(f"  {dt}: {','.join(pk)}")
