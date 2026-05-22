# 期货求魔系统 — 通达信主图叠加指标

来源：bilibili UP「期货求魔」9 集教学。完整方法论见 [`METHOD.md`](../METHOD.md)，Python 实现见 [`src/indicators/qiumo.py`](../src/indicators/qiumo.py)。

## 导入步骤

1. 通达信菜单 **工具** → **公式管理器**（快捷键 `Ctrl+F`）
2. 左侧选 **技术指标** → 你的分类目录 → **新建**
3. 公式名称：`QIUMO`，类型选 **主图叠加**
4. 把下方公式整段粘进右侧编辑框 → **测试公式** → **确定**
5. 回到 K 线图右键 **加载指标** 选 `QIUMO`

## 公式

```
{
  期货求魔交易系统 — 通达信主图叠加指标
  建议周期: 5 分钟 K 线 (3 分钟也可, 但参数需调整)
}

{ ===== 三条均线 ===== }
MA20:   MA(C, 20),  COLORWHITE,  LINETHICK1;
MA120:  MA(C, 120), COLORYELLOW, LINETHICK1;
MA250:  MA(C, 250), COLORMAGENTA,LINETHICK2;

{ ===== MA250 斜率与趋势状态 ===== }
{ 注: 不能用 SLOPE 作变量名 (通达信已有同名内建函数) }
SLP_N := 10;
SLP   := (MA250 - REF(MA250, SLP_N)) / C;
SLP_E := 0.0001;
UPT  := SLP > SLP_E;
DNT  := SLP < -SLP_E;
RNG  := NOT(UPT) AND NOT(DNT);

{ ===== Pivot 检测 (左右各 K 根的极值; K 根后确认) ===== }
K := 3;
PL := REF(LOW, K)  = LLV(LOW, 2*K+1);
PH := REF(HIGH, K) = HHV(HIGH, 2*K+1);

{ ===== 靠近 MA250 的 pivot ===== }
NEAR_PCT := 0.3;
NL := PL AND ABS(REF(LOW, K)  - REF(MA250, K)) / REF(MA250, K) * 100 <= NEAR_PCT;
NH := PH AND ABS(REF(HIGH, K) - REF(MA250, K)) / REF(MA250, K) * 100 <= NEAR_PCT;

{ ===== 最近两次"靠近 MA250 的 pivot"的价位 ===== }
LAST_NL := VALUEWHEN(1, NL, REF(LOW, K));
PREV_NL := VALUEWHEN(2, NL, REF(LOW, K));
LAST_NH := VALUEWHEN(1, NH, REF(HIGH, K));
PREV_NH := VALUEWHEN(2, NH, REF(HIGH, K));

{ ===== 入场信号 ===== }
LONG_S2  := NL AND LAST_NL > PREV_NL AND C > MA250 AND (UPT OR RNG);
SHORT_S2 := NH AND LAST_NH < PREV_NH AND C < MA250 AND (DNT OR RNG);
LONG_S1  := NL AND LAST_NL >= PREV_NL AND C > HHV(HIGH, 7) AND (UPT OR RNG);
SHORT_S1 := NH AND LAST_NH <= PREV_NH AND C < LLV(LOW, 7)  AND (DNT OR RNG);

{ 不用 OPEN 前缀, 避免与开盘价关键字混淆 }
BUY_SIG  := LONG_S1 OR LONG_S2;
SELL_SIG := SHORT_S1 OR SHORT_S2;

{ ===== 出场信号 ===== }
EXIT_LONG_WARN  := CROSS(MA120, C);
EXIT_LONG_FULL  := CROSS(MA250, C);
EXIT_SHORT_WARN := CROSS(C, MA120);
EXIT_SHORT_FULL := CROSS(C, MA250);

{ ===== 价差极端 ===== }
EXT_N := 1200;
SPD := C - MA250;
EXT_L  := ABS(SPD) = HHV(ABS(SPD), EXT_N) AND SPD > 0;
EXT_S  := ABS(SPD) = HHV(ABS(SPD), EXT_N) AND SPD < 0;

{ ===== 主图标记 ===== }
DRAWICON(BUY_SIG,          LOW * 0.998,  1);
DRAWICON(SELL_SIG,         HIGH * 1.002, 2);
DRAWICON(EXIT_LONG_FULL,   HIGH * 1.002, 3);
DRAWICON(EXIT_SHORT_FULL,  LOW * 0.998,  3);
DRAWICON(EXIT_LONG_WARN,   HIGH * 1.002, 8);
DRAWICON(EXIT_SHORT_WARN,  LOW * 0.998,  8);

DRAWTEXT(EXT_L, HIGH * 1.005, '极端拉升 全部止盈'), COLORYELLOW;
DRAWTEXT(EXT_S, LOW  * 0.995, '极端杀跌 全部止盈'), COLORYELLOW;

{ ===== 趋势染色 ===== }
STICKLINE(UPT, LOW,  LOW  * 0.998, 1.5, 0), COLORRED;
STICKLINE(DNT, HIGH, HIGH * 1.002, 1.5, 0), COLORGREEN;
```

## 图例

| 标记 | 含义 |
| --- | --- |
| 白线 | MA20（快/出场参考） |
| 黄线 | MA120（中/部分止盈参考） |
| 紫线（粗） | MA250（慢/趋势锚） |
| K 线下方淡红柱 | 多头势（MA250 斜率向上） |
| K 线上方淡绿柱 | 空头势（MA250 斜率向下） |
| 上箭头（下方） | 多头入场 |
| 下箭头（上方） | 空头入场 |
| 叉号 | 全平（MA250 被有效穿越） |
| 小圆点 | 减仓警告（MA120 被有效穿越，需手动减仓） |
| 黄字 | 价差极端（按方法论该全部止盈） |

## 调参

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `SLP_N` | 10 | MA250 斜率回归窗口 |
| `SLP_E` | 0.0001 | 斜率阈值，越大越严格 |
| `K` | 3 | swing 半宽（波动大的品种调到 5） |
| `NEAR_PCT` | 0.3 | 视为靠近 MA250 的百分比距离 |
| `EXT_N` | 1200 | 价差极端窗口（5min ≈ 1 周；12000 ≈ 1 年） |

## 与 Python 版的差异

- **价差排名**：TDX 没有 `rank(pct=True)`，用 `HHV(ABS(spread), N)` 等于当前值近似，只在打破新高极端才触发，比 Python 稀疏
- **W 底检测**：简化为"靠近底 + 高低相等以上 + 突破最近 7 根高"，没有完整的 interim_high 状态机
- **部分平仓**：只能"信号→提示"，不能控仓位，MA120 跌破只画圆点提示
- **多周期对照**：本公式只看当前周期
- **止损"前低 + 1-2 跳"缓冲**：bar 内挂单的事情，主图公式表达不了
