"""
版本A(趋势确认) 日线 + 高波筛选 的样本外验证

做法：每个品种按时间 50/50 切分为 样本内(IS) / 样本外(OOS)。
- 选品规则用"样本内波动率"(结构性、可提前知道、可实现)。
- 在样本外(完全未用于选品)上度量策略表现，检验"日线+高波"是否有持续 edge。

策略：版本A —— 0轴下方金叉关注; LON 上穿零轴买入; 死叉平仓(做空镜像)。
数据：data_futures/15min/ 合成日线(每16根15min)。手续费万3。
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lon_futures_backtest import DATA_DIR, MANIFEST, COMMISSION
from lon_long_short_backtest import _simulate, _metrics, _trade_stats
from lon_trend_backtest import build_targets_trend, _agg

PPY = 252
VOL_HI = 0.30  # 高波阈值(年化)


def seg_eval(raw_seg: pd.DataFrame):
    """对一段 15min 数据：合成日线 -> 跑版本A -> 返回(收益, 夏普, 波动, 交易, 买入持有)。"""
    d = _agg(raw_seg, 16)
    if len(d) < 60:
        return None
    c = d['close'].values
    tgt = build_targets_trend(d)
    sim = _simulate(tgt, c, commission=COMMISSION)
    ret, dd, sh = _metrics(sim['ret'], sim['equity'], PPY)
    tr, win = _trade_stats(tgt, sim['equity'])
    vol = np.diff(np.log(raw_seg['close'].values)).std() * np.sqrt(252 * 16)
    bh = float(c[-1] / c[0] - 1.0)
    return dict(ret=ret, sharpe=sh, vol=vol, trades=tr, win=win, bh=bh)


def main():
    man = pd.read_csv(MANIFEST).set_index('product')
    rows = []
    for f in sorted(os.listdir(DATA_DIR)):
        if not f.endswith('.parquet'):
            continue
        p = f[:-8]
        raw = pd.read_parquet(os.path.join(DATA_DIR, f)).rename(columns={'datetime': 'date'})
        if len(raw) < 400 or raw['volume'].sum() <= 0:
            continue
        raw = raw.sort_values('date').reset_index(drop=True)
        mid = len(raw) // 2
        IS = seg_eval(raw.iloc[:mid]); OOS = seg_eval(raw.iloc[mid:])
        if IS is None or OOS is None:
            continue
        nm = str(man.loc[p, 'name']) if p in man.index else p
        rows.append(dict(p=p, nm=nm, is_vol=IS['vol'], is_ret=IS['ret'],
                         oos_ret=OOS['ret'], oos_sh=OOS['sharpe'], oos_bh=OOS['bh'],
                         oos_tr=OOS['trades'], oos_win=OOS['win']))
    d = pd.DataFrame(rows)

    def stat(name, sub):
        o = sub['oos_ret'].values; bh = sub['oos_bh'].values
        print(f"  {name:<24} n={len(sub):2}  OOS均{o.mean()*100:+6.1f}%  中位{np.median(o)*100:+6.1f}%  "
              f"正收益{int((o>0).sum()):2}/{len(sub)}  跑赢B&H{int((o>bh).sum()):2}/{len(sub)}  夏普均{sub['oos_sh'].mean():+.2f}")

    print(f"\n{'='*92}\n版本A 日线 — 样本外(OOS)验证   (按样本内IS波动率选品, 阈值{VOL_HI:.0%})\n{'='*92}")
    print("【整体 OOS】")
    stat("全部品种", d)
    print("\n【按样本内IS波动率筛选 -> 看样本外OOS表现】")
    hi = d[d.is_vol >= VOL_HI]; mid_ = d[(d.is_vol >= 0.15) & (d.is_vol < VOL_HI)]; lo = d[d.is_vol < 0.15]
    stat("高波篮子(IS_vol>=30%)", hi)
    stat("中波篮子(15-30%)", mid_)
    stat("低波篮子(<15%)", lo)

    print("\n【参考：高波篮子 IS 内表现 vs OOS 表现】")
    print(f"  高波篮子 样本内IS均收益 {hi['is_ret'].mean()*100:+.1f}%   ->   样本外OOS均收益 {hi['oos_ret'].mean()*100:+.1f}%")

    print("\n【高波篮子逐品种 OOS】(按 OOS 收益排序)")
    print(f"  {'品种':<5}{'名称':<8}{'IS波动':>7}{'IS收益':>9}{'OOS收益':>9}{'OOS夏普':>8}{'OOS胜率':>8}{'交易':>5}")
    for _, r in hi.sort_values('oos_ret', ascending=False).iterrows():
        print(f"  {r['p']:<5}{r['nm'][:7]:<8}{r['is_vol']*100:>6.0f}%{r['is_ret']*100:>8.1f}%"
              f"{r['oos_ret']*100:>8.1f}%{r['oos_sh']:>8.2f}{r['oos_win']*100:>7.0f}%{int(r['oos_tr']):>5}")

    # 持续性：IS与OOS收益相关性
    print(f"\n【持续性】corr(IS收益, OOS收益) 全样本 = {d['is_ret'].corr(d['oos_ret']):+.3f}")
    print(f"          corr(IS波动, OOS收益) 全样本 = {d['is_vol'].corr(d['oos_ret']):+.3f}")


if __name__ == "__main__":
    main()
