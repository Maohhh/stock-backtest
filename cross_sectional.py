"""验证: (1)当前指标核心是同一信号  (2)截面相对价值反转 vs 时序反转, 在11干净品种上。

时序信号本质 = (C-MA_N)/std_N 的偏离反转(N不同只是皮)。
截面信号 = 同一时点跨品种的相对错位反转(去市场/趋势, 利用品种独立性)。
"""
import numpy as np, pandas as pd
import strategy_range as sr

CLEAN = ["V","PX","MA","SI","ZN","SH","BU","RM","UR","PG","AP"]  # 干净可交易回归型

# ---------- (1) 核心同源性: 不同窗口的偏离信号相关 ----------
print("="*72)
print("(1) 当前指标核心同源性: 偏离信号 dev_N=(C-MA_N)/std_N 两窗口相关")
print("="*72)
cors=[]
for p in CLEAN:
    c=sr.DATA[p]["close"]
    dev100=((c-c.rolling(100).mean())/c.rolling(100).std())
    dev250=((c-c.rolling(250).mean())/c.rolling(250).std())
    cors.append(dev100.corr(dev250))
print(f"dev100 vs dev250 相关: 中位={np.median(cors):.2f} (高=同一信号不同窗口)")

# ---------- 日频 panel ----------
def daily_close(df):
    s=pd.Series(df["close"].values, index=pd.to_datetime(df["datetime"]))
    return s.groupby(s.index.normalize()).last()
panel=pd.DataFrame({p:daily_close(sr.DATA[p]) for p in CLEAN}).sort_index()
panel=panel.dropna()
ret=panel.pct_change()
print(f"\n日频panel: {panel.shape[0]}个交易日 × {panel.shape[1]}品种 "
      f"({panel.index[0].date()}~{panel.index[-1].date()})")

def perf(r, cost_turn=None, label=""):
    r=r.dropna()
    sh=r.mean()/r.std()*np.sqrt(252) if r.std()>0 else 0
    days=(r.index[-1]-r.index[0]).days
    ann=(1+r).prod()**(365/days)-1
    mdd=((1+r).cumprod()/(1+r).cumprod().cummax()-1).min()
    segs=np.array_split(np.arange(len(r)),5)
    ss=[r.iloc[ix].mean()/r.iloc[ix].std()*np.sqrt(252) if r.iloc[ix].std()>0 else 0 for ix in segs]
    print(f"{label:<26} 夏普={sh:>5.2f} 年化={ann:>+6.1%} 回撤={mdd:>6.1%} "
          f"5段=[{' '.join(f'{s:+.1f}' for s in ss)}] 最差={min(ss):+.1f}")
    return sh

# ---------- (2) 截面短期反转: 做多近期最弱, 做空近期最强(dollar-neutral) ----------
print("="*72)
print("(2) 截面相对价值反转 (日频, 多空中性, 毛收益/含成本)")
print("="*72)
for L in [1,2,3,5,10]:
    mom=panel/panel.shift(L)-1                 # 过去L日收益
    x=mom.sub(mom.mean(axis=1),axis=0)         # 截面去均值(去市场/趋势)
    w=-x.div(x.abs().sum(axis=1),axis=0)       # 反转, gross=1 多空
    pnl=(w.shift(1)*ret).sum(axis=1)           # 用昨信号今收益(无前视)
    turn=(w-w.shift(1)).abs().sum(axis=1)
    gross=perf(pnl, label=f"  L={L} 毛(0bp)")
    for cost in [0.0005,0.001]:
        net=pnl-turn.shift(1).fillna(0)*cost
        perf(net, label=f"  L={L} 净({cost*1e4:.0f}bp)")
    print()

print("="*72)
print("(3) 对症降换手: 信号平滑(EWMA持有) + 死区(只在错位够大时调仓) — L=5基础")
print("="*72)
L=5
mom=panel/panel.shift(L)-1
x=mom.sub(mom.mean(axis=1),axis=0)
xz=x.div(x.std(axis=1),axis=0)                      # 截面z
raw=-xz.div(xz.abs().sum(axis=1),axis=0)            # 目标权重(反转)

def run_w(w,label):
    pnl=(w.shift(1)*ret).sum(axis=1)
    turn=(w-w.shift(1)).abs().sum(axis=1)
    for cost in [0.0005,0.001]:
        net=pnl-turn.shift(1).fillna(0)*cost
        sh=net.mean()/net.std()*np.sqrt(252) if net.std()>0 else 0
        days=(net.index[-1]-net.index[0]).days; ann=(1+net).prod()**(365/days)-1
        segs=np.array_split(np.arange(len(net.dropna())),5)
        nd=net.dropna()
        ss=[nd.iloc[ix].mean()/nd.iloc[ix].std()*np.sqrt(252) if nd.iloc[ix].std()>0 else 0 for ix in segs]
        at=turn.mean()
        print(f"{label} 净{cost*1e4:.0f}bp: 夏普={sh:>5.2f} 年化={ann:>+6.1%} 日均换手={at:.2f} 5段最差={min(ss):+.1f}")

# EWMA平滑(不同半衰期=拉长持有)
for hl in [3,5,10,20]:
    w=raw.ewm(halflife=hl).mean()
    w=w.div(w.abs().sum(axis=1),axis=0)
    run_w(w,f"  EWMA-hl{hl:<2}")
print()
# 死区: 仅保留|z|>阈值的腿, 弱信号不持仓(降换手+只押强错位)
for hl,th in [(10,0.5),(10,1.0),(20,0.5),(20,1.0)]:
    w0=raw.where(xz.abs()>=th,0.0)
    w=w0.ewm(halflife=hl).mean()
    s=w.abs().sum(axis=1); w=w.div(s.where(s>0,1),axis=0)
    run_w(w,f"  死区z>{th} +EWMA-hl{hl:<2}")
