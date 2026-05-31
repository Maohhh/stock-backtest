import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import strategy_range as sr
import equity_curve as ec   # 复用sim

CLEAN9=["V","PX","MA","SH","BU","RM","UR","PG","AP"]
panel=pd.DataFrame({p:ec.sim(sr.DATA[p],100,2.0,3.0) for p in CLEAN9}).fillna(0).sort_index()
port=panel.mean(axis=1).dropna()
cap0=100000

fig,axes=plt.subplots(2,1,figsize=(12,8),gridspec_kw={"height_ratios":[3,1]},sharex=True)
# 资金曲线: 1x/2x/3x 名义
for lev,c,lab in [(1,"#1f77b4","1x名义(年化19%/回撤8%)"),
                  (2,"#ff7f0e","2x名义(年化41%/回撤16%)"),
                  (3,"#d62728","3x名义(年化64%/回撤23%)")]:
    eq=cap0*(1+port*lev).cumprod()
    axes[0].plot(eq.index,eq.values,color=c,lw=1.6,label=lab)
axes[0].axhline(cap0,color="gray",ls="--",lw=0.8)
axes[0].set_yscale("log")
axes[0].set_ylabel("账户资金(元, 对数轴)")
axes[0].set_title("9品种区间反转组合 · 10万本金资金曲线 (真实成本2.5bp/边, 2023-09~2026-05)")
axes[0].legend(loc="upper left",fontsize=9)
axes[0].grid(True,alpha=0.3)
# 回撤(1x)
eq1=(1+port).cumprod(); dd=(eq1/eq1.cummax()-1)
axes[1].fill_between(dd.index,dd.values*100,0,color="#1f77b4",alpha=0.4)
axes[1].set_ylabel("回撤 %(1x)")
axes[1].set_xlabel("日期")
axes[1].grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig("data_futures/backtest_qiumo/equity_curve.png",dpi=110)
print("saved equity_curve.png")
print(f"1x最终: {cap0*eq1.iloc[-1]:,.0f}  最大回撤{dd.min()*100:.1f}%")
