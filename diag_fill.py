"""诚实性补充: 反转入场在2σ极端位的成交质量(逆选择风险)。

均值回归在下轨买=接下跌的刀。若用限价单在轨上挂, 只有继续下跌才成交(逆选择);
若用市价单, 付价差。检验: 入场后下一根的即时方向, 估真实滑点是否>半tick。
"""
import glob,numpy as np,pandas as pd
import strategy_range as sr
TICK={"V":1,"PX":2,"MA":1,"SI":5,"ZN":5,"SH":1,"BU":1,"RM":1,"UR":1,"PG":1,"AP":1}
CLEAN=list(TICK)

# 统计: 触轨入场后, 当根->下根的价格继续不利移动幅度(逆选择代理)
adverse=[];favor=[]
for p in CLEAN:
    df=sr.DATA[p]; C=df["close"];N=100;K=2.0
    mid=C.rolling(N).mean();sd=C.rolling(N).std()
    lo=(mid-K*sd);up=(mid+K*sd);c=C.to_numpy()
    lon=(c<lo.to_numpy()); shr=(c>up.to_numpy())
    nxt=np.r_[c[1:]/c[:-1]-1,np.nan]
    tickp=TICK[p]/np.median(c)
    # 多头入场后下一根收益(负=继续下跌=逆选择), 以tick为单位
    a=nxt[lon]/tickp; a=a[~np.isnan(a)]
    s=-nxt[shr]/tickp; s=s[~np.isnan(s)]
    adverse.append(np.concatenate([a,s]))
allv=np.concatenate(adverse)
print(f"触轨入场后下一根的方向(以tick计, 正=有利立刻反弹, 负=继续不利):")
print(f"  中位={np.median(allv):+.2f}tick 均值={np.mean(allv):+.2f}tick  立即反弹比例={(allv>0).mean():.0%}")
print(f"  含义: 若>0占多数, 反转入场点位质量好, 不存在严重逆选择")

# 用5min数据看入场当根的真实可成交范围(15min信号但执行)?  -- 用high-low/tick 估计单根内可滑点空间
