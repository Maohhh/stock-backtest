# 开盘区间突破 (ORB) —— 极简顺势短线指标

代码: `src/indicators/opening_range_breakout.py`，回测: `backtest_orb.py`。
这是在"先想清楚短线到底怎么做"之后, 重做的**第二版**指标 —— 从复杂高胜率反转,
转向**极简、低胜率高盈亏比的顺势突破**。

## 一、为什么重做(基于研究的反思)

上网调研后的三条结论(来源见文末):

1. **简单 > 复杂**: 复杂策略是把历史背下来, 规则越多越脆、实盘必崩; 简单规则捕捉的是
   基本的市场机制与人性, 抗过拟合、且人能稳定执行。第一版那套复合反转正是反面教材。
2. **"高胜率"很可能是错误目标**: 顶尖交易者常年胜率仅 35–50%, 靠的是 2:1~4:1 的盈亏比
   和"让利润奔跑"。"胜率不重要, 期望才重要。" 纯靠 K 线追求高胜率, 恰是最难的路。
3. **唯一有大样本实证的简单系统化短线 edge 是开盘区间突破(ORB)**:
   Zarattini/Barbon/Aziz(2016–2023, 7000+ 美股)显示, 限定"当日放量异动"标的的 5min ORB
   有显著正期望(夏普~2.8), 扣成本后依然成立。ES/NQ 等指数期货是其经典战场。

## 二、指标规则(只有 4 条)

1. **开盘区间**: 每个交易会话(国内期货日盘/夜盘各算一个, 按 K 线间隔自动切分)前 N 根
   K 线的最高/最低 = 区间上下沿(默认 `or_bars=2`, 即 30min)。
2. **突破**: 收盘突破区间上沿做多 / 下沿做空。
3. **方向过滤**: 突破方向须与会话 VWAP 一致(站上 VWAP 才做多)。
4. **放量过滤**: 突破那根相对成交量 `RVOL ≥ rvol_min`(默认 1.2), 只在"有行情"时出手。

止损 = 开盘区间另一侧; 利润让它奔跑(回测以**会话末收盘**或**ATR 跟踪止损**了结)。
每会话每方向只取首次突破。无未来函数(含单测验证)。

```python
from src.indicators import opening_range_breakout
sig = opening_range_breakout(df)          # df 需 o/h/l/c/v + 时间戳
entries_long = df[sig['long_signal']]
```

## 三、回测结果(诚实版, 按板块)

加权连续 15min, 2023-09 ~ 2026-05, 全板块代表品种, 收益以 R(进场到区间止损)计。
**ORB 是低胜率高盈亏比结构, 看盈利因子>1 与正期望, 不看胜率。**

**A. 市价单进出(单边滑点 1 tick, 现实偏保守), 各板块期望(R):**

| 板块 | 交易数 | 胜率 | 期望(R) |
|---|---|---|---|
| **股指(IF/IH/IC/IM)** | 2004 | 47% | **+0.007** ✅ |
| 黑色/工业品 | 3018 | 39% | −0.089 |
| 有色 | 2620 | 44% | −0.077 |
| 能化 | 3070 | 40% | −0.104 |
| 农产品 | 3660 | 36% | −0.132 |
| 贵金属 | 2103 | 45% | −0.195(沪银 AG 拖累) |

股指内部: IM 中证1000 期望 **+0.041 / 盈利因子 1.19**, IC +0.008, IF +0.007, IH −0.026。

**B. 被动限价成交(滑点 0)各板块期望(R):**

| 板块 | 期望(R) | 板块 | 期望(R) |
|---|---|---|---|
| 黑色/工业品 | +0.027 ✅ | 股指 | +0.019 ✅ |
| 有色 | +0.013 ✅ | 能化 | −0.004 |
| 农产品 | −0.020 | 贵金属 | −0.007 |

**全市场汇总: 市价单 −0.101R, 被动成交 +0.003R(由负转正)。**

## 四、结论

1. **结构正确**: 胜率 36–52%(低)、盈亏比多在 1.0–1.4(高), 正是顺势突破该有的样子,
   远比第一版反转稳健, 规则也简单得多(4 条)。
2. **存在真实毛 edge**: 被动成交下全市场期望转正(+0.003R), 说明信号本身有顺势优势。
3. **股指是最顺的战场**: IF/IH/IC/IM 即便用市价单(含滑点)也接近/越过盈亏平衡, IM 最强 ——
   与海外 ES/NQ 是 ORB 经典标的完全一致(指数日内趋势性强、波动相对成本足够大)。
4. **成败仍在执行**: 同一套信号从市价单切到被动限价, 全市场期望从 −0.10 跳到 +0.003。
   **短线真正的对手是滑点与执行, 不是方向。** 农产品/沪银因波动相对成本太小而难赚。

> 研究性结论, 非可直接下单系统。下一步若走实盘: 锁定股指(尤其 IM)做单品种参数优化 +
> walk-forward + 真实成交建模 + 仓位风控。盲目调参做"漂亮回测" = 过拟合。

## 五、复现

```bash
python backtest_orb.py                       # 全板块, 市价单/滑点1tick, 让利润跑到会话末
python backtest_orb.py --slippage-ticks 0    # 被动限价(毛 edge 天花板)
python backtest_orb.py --trail-atr 2.0       # 启用 ATR 跟踪止损
python backtest_orb.py --products IF,IH,IC,IM --or-bars 1   # 只看股指, 15min开盘区间
python -m unittest tests.test_indicators     # 指标单测(含无未来函数)
```

## 参考来源

- Zarattini, Barbon & Aziz, *A Profitable Day Trading Strategy For The U.S. Equity Market* — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4729284
- Why simple quant rules outperform — https://www.quantifiedstrategies.com/why-simple-quant-trading-rules-often-outperform-complicated-systems/
- Win rate vs expectancy — https://www.tradezella.com/blog/win-rate
- Order flow scalping(为何纯 K 线做不了高胜率 scalp) — https://bookmap.com/blog/can-real-time-order-flow-give-you-an-edge-in-scalp-trading
- Opening range breakout 实现要点 — https://highstrike.com/opening-range/
