# 期货求魔系统 — 通达信主图叠加指标

来源：bilibili UP「期货求魔」9 集教学。完整方法论见 [`METHOD.md`](../METHOD.md)，Python 实现见 [`src/indicators/qiumo.py`](../src/indicators/qiumo.py)。

## 导入步骤

1. 通达信菜单 **工具** → **公式管理器**（快捷键 `Ctrl+F`）
2. 左侧选 **技术指标** → 你的分类目录 → **新建**
3. 公式名称：`QIUMO`，类型选 **主图叠加**
4. 把下方公式整段粘进右侧编辑框 → **测试公式** → **确定**
5. 回到 K 线图右键 **加载指标** 选 `QIUMO`

## 公式（v5，放宽入场 + 仓位自动出局 + 诊断点）

```
{
  期货求魔交易系统 - 通达信主图叠加指标
  建议周期: 5 分钟 K 线
}

{ ===== 三条均线 ===== }
MA20:   MA(C, 20),  COLORWHITE,  LINETHICK1;
MA120:  MA(C, 120), COLORYELLOW, LINETHICK1;
MA250:  MA(C, 250), COLORMAGENTA,LINETHICK2;

{ ===== MA250 斜率与趋势状态 ===== }
SLP_N := 10;
SLP   := (MA250 - REF(MA250, SLP_N)) / C;
SLP_E := 0.0001;
UPT := SLP > SLP_E;
DNT := SLP < -SLP_E;
RNG := NOT(UPT) AND NOT(DNT);

{ ===== Pivot 检测 ===== }
K := 3;
PL := REF(LOW, K)  = LLV(LOW, 2*K+1);
PH := REF(HIGH, K) = HHV(HIGH, 2*K+1);

{ ===== 靠近 MA250 的 pivot ===== }
NEARP := 0.3;
NL := PL AND ABS(REF(LOW, K)  - REF(MA250, K)) / REF(MA250, K) * 100 <= NEARP;
NH := PH AND ABS(REF(HIGH, K) - REF(MA250, K)) / REF(MA250, K) * 100 <= NEARP;

{ ===== 入场信号 (STRICT_MODE = 1 启用突破确认) ===== }
STRICT_MODE := 0;

BUY_BASE_LOOSE  := NL AND C > MA250 AND (UPT OR RNG);
BUY_BASE_STRICT := BUY_BASE_LOOSE AND C > HHV(HIGH, 7);
BUY_BASE := (STRICT_MODE = 1 AND BUY_BASE_STRICT) OR (STRICT_MODE = 0 AND BUY_BASE_LOOSE);

SELL_BASE_LOOSE  := NH AND C < MA250 AND (DNT OR RNG);
SELL_BASE_STRICT := SELL_BASE_LOOSE AND C < LLV(LOW, 7);
SELL_BASE := (STRICT_MODE = 1 AND SELL_BASE_STRICT) OR (STRICT_MODE = 0 AND SELL_BASE_LOOSE);

{ ===== 出场基本事件 ===== }
X_LONG_WARN  := CROSS(MA120, C);
X_LONG_FULL  := CROSS(MA250, C);
X_SHORT_WARN := CROSS(C, MA120);
X_SHORT_FULL := CROSS(C, MA250);

{ ===== 仓位状态代理 (反向信号 + 最长 200 根持仓自动出局) ===== }
MAX_HOLD := 200;

LAST_BUY := BARSLAST(BUY_BASE);
LAST_END_LONG := MIN(BARSLAST(X_LONG_FULL), BARSLAST(SELL_BASE));
IN_LONG := LAST_BUY < LAST_END_LONG AND LAST_BUY <= MAX_HOLD;

LAST_SELL := BARSLAST(SELL_BASE);
LAST_END_SHORT := MIN(BARSLAST(X_SHORT_FULL), BARSLAST(BUY_BASE));
IN_SHORT := LAST_SELL < LAST_END_SHORT AND LAST_SELL <= MAX_HOLD;

{ ===== 经持仓状态过滤的出场信号 ===== }
EXIT_LONG_WARN  := X_LONG_WARN  AND IN_LONG;
EXIT_LONG_FULL  := X_LONG_FULL  AND IN_LONG;
EXIT_SHORT_WARN := X_SHORT_WARN AND IN_SHORT;
EXIT_SHORT_FULL := X_SHORT_FULL AND IN_SHORT;

{ ===== 价差极端 ===== }
EXT_N := 1200;
SPD := C - MA250;
EXT_L := ABS(SPD) = HHV(ABS(SPD), EXT_N) AND SPD > 0 AND IN_LONG;
EXT_S := ABS(SPD) = HHV(ABS(SPD), EXT_N) AND SPD < 0 AND IN_SHORT;

{ ===== 诊断点 ===== }
DRAWTEXT(NL, LOW  * 0.997, '·'), COLORMAGENTA;
DRAWTEXT(NH, HIGH * 1.003, '·'), COLORMAGENTA;

{ ===== 主图标记 ===== }
DRAWTEXT(BUY_BASE,         LOW  * 0.995, '买'),   COLORYELLOW;
DRAWTEXT(SELL_BASE,        HIGH * 1.005, '卖'),   COLORYELLOW;
DRAWTEXT(EXIT_LONG_FULL,   HIGH * 1.002, '平多'), COLORCYAN;
DRAWTEXT(EXIT_SHORT_FULL,  LOW  * 0.998, '平空'), COLORCYAN;
DRAWTEXT(EXIT_LONG_WARN,   HIGH * 1.002, '减'),   COLORWHITE;
DRAWTEXT(EXIT_SHORT_WARN,  LOW  * 0.998, '减'),   COLORWHITE;
DRAWTEXT(EXT_L,            HIGH * 1.005, '极'),   COLORCYAN;
DRAWTEXT(EXT_S,            LOW  * 0.995, '极'),   COLORCYAN;

{ ===== 趋势染色 ===== }
STICKLINE(UPT, LOW,  LOW  * 0.998, 1.5, 0), COLORRED;
STICKLINE(DNT, HIGH, HIGH * 1.002, 1.5, 0), COLORGREEN;
```

## 图例

| 标记 | 颜色 | 含义 |
| --- | --- | --- |
| 白线 | 白 | MA20 |
| 黄线 | 黄 | MA120 |
| 紫线（粗） | 紫 | MA250 |
| K 线下淡红柱 | 红 | 多头势 |
| K 线上淡绿柱 | 绿 | 空头势 |
| `·` 紫点 | 紫 | 系统识别的回踩位（NL/NH） |
| `买` | 黄 | 多头入场 |
| `卖` | 黄 | 空头入场 |
| `平多` `平空` | 青 | 全平（MA250 被穿越） |
| `减` | 白 | 减仓警告（MA120 被穿） |
| `极` | 青 | 价差极端，按方法论全部止盈 |

## 用法和诊断流程

1. **先看紫色 · 点**：系统认为这里"价格回踩到 MA250 附近形成 swing"。点越多代表 NEARP/K 设得越宽。如果你的图上**几乎没有紫点**——说明 `NEARP=0.3` 对你品种太严，加到 0.5 或 1.0。
2. **再看黄色 买/卖**：紫点 + 收盘已经站上/跌破 MA250 + 趋势确认才会触发。
3. **青色 平多/平空** 只在系统估计你"还在仓位中"时显示。如果你只看到 减 没看到 买/卖，问题往往是你只能看可视范围，而入场信号在更左边。

## STRICT_MODE

公式头部 `STRICT_MODE := 0` 是默认宽松模式（只要 NL + 站上 MA250）。改成 `1` 启用严格模式（再加上"收盘破 7 根 K 线高"）——信号会少一半但每次更"硬"。

## 调参

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `STRICT_MODE` | 0 | 1 = 严格入场（再加突破确认）|
| `SLP_N` | 10 | MA250 斜率回归窗口 |
| `SLP_E` | 0.0001 | 斜率阈值，越大越严格 |
| `K` | 3 | swing 半宽 |
| `NEARP` | 0.3 | 视为靠近 MA250 的百分比距离 |
| `MAX_HOLD` | 200 | 仓位状态最长保持根数（200×5min ≈ 4 个交易日）|
| `EXT_N` | 1200 | 价差极端窗口 |

## 与 Python 版的差异（已知简化）

- **价差排名**：TDX 没有 `rank(pct=True)`，用 `HHV(ABS(spread), N)` 等于当前值近似
- **入场信号**：Python 版区分 W 底 / 抬高底；通达信合成一个条件
- **部分平仓**：只能"信号→提示"
- **仓位状态**：BARSLAST 代理，与实盘不完全一致
- **多周期对照**：不支持
- **止损"前低 + 1-2 跳"缓冲**：不支持

## 修订记录

| 版本 | 问题 → 修复 |
| --- | --- |
| v1→v2 | `SLOPE` 与内建函数冲突 → 改 `SLP` |
| v2→v3 | 3 参 `VALUEWHEN` 不是 TDX 原生 → 简化为 "突破 7 根高" |
| v3→v4 | DRAWICON 1/3 显示为 "B"/"空" → 改 DRAWTEXT 明文；MA250 上下穿太频繁 → 加 IN_LONG/IN_SHORT 代理 |
| v4→v5 | 严格入场太少触发 + IN_LONG 单边趋势下永远 TRUE → 放宽默认入场、加 MAX_HOLD 自动出局、加 NL/NH 诊断点 |
