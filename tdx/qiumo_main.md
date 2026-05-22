# 期货求魔系统 — 通达信主图叠加指标

来源：bilibili UP「期货求魔」9 集教学。完整方法论见 [`METHOD.md`](../METHOD.md)，Python 实现见 [`src/indicators/qiumo.py`](../src/indicators/qiumo.py)。

## v6 严格按讲解原文实现的入场逻辑

5_03 原文里清晰区分了两类入场：

| 信号 | 讲解原文 | 代码体现 |
| --- | --- | --- |
| Signal 1 理想型 | 突破 MA250 → 回踩到 MA250 **守住没破** → 突破前期高点 | `NL_HELD AND C > PRE_HIGH` |
| Signal 2 洗盘型 | 突破 MA250 → 回踩**跌破 MA250（假突破）**→ 重新站上 → 突破前期高点 | `NL_BROKE AND C > MA250 AND C > PRE_HIGH` |

两个信号共同要求：
- 回踩点靠近 MA250（`NEARP` 控制距离百分比）
- **收盘突破前期高点**（`PRE_HIGH` = 回踩之前 20 根 K 线的最高高）

5_04 的三档移动平仓：

| 出场触发 | 含义 | 代码体现 |
| --- | --- | --- |
| 价差极端 | 价格离 MA250 偏离达近 N 根最大 → 全部止盈 | `EXT_L` / `EXT_S` |
| MA120 跌破 | 减仓警告（保留小底仓）| `EXIT_LONG_WARN` / `EXIT_SHORT_WARN` |
| MA250 跌破 | 全部止盈/止损 | `EXIT_LONG_FULL` / `EXIT_SHORT_FULL` |

做空全部镜像。

## 导入步骤

1. 通达信菜单 **工具** → **公式管理器**（快捷键 `Ctrl+F`）
2. 左侧选 **技术指标** → 你的分类目录 → **新建**
3. 公式名称：`QIUMO`，类型选 **主图叠加**
4. 把下方公式整段粘进右侧编辑框 → **测试公式** → **确定**
5. 回到 K 线图右键 **加载指标** 选 `QIUMO`

## 公式

```
{
  期货求魔交易系统 - 通达信主图叠加指标 (v6)
  建议周期: 5 分钟 K 线
}

{ ===== 三条均线 ===== }
MA20:   MA(C, 20),  COLORWHITE,  LINETHICK1;
MA120:  MA(C, 120), COLORYELLOW, LINETHICK1;
MA250:  MA(C, 250), COLORMAGENTA,LINETHICK2;

{ ===== MA250 斜率 ===== }
SLP_N := 10;
SLP   := (MA250 - REF(MA250, SLP_N)) / C;
SLP_E := 0.0001;
UPT := SLP > SLP_E;
DNT := SLP < -SLP_E;
RNG := NOT(UPT) AND NOT(DNT);

{ ===== Pivot 检测 ===== }
K := 3;
NEARP := 0.3;
PL := REF(LOW, K)  = LLV(LOW, 2*K+1);
PH := REF(HIGH, K) = HHV(HIGH, 2*K+1);
NL := PL AND ABS(REF(LOW, K)  - REF(MA250, K)) / REF(MA250, K) * 100 <= NEARP;
NH := PH AND ABS(REF(HIGH, K) - REF(MA250, K)) / REF(MA250, K) * 100 <= NEARP;

{ ===== 回踩类型 (理想型 vs 洗盘型) ===== }
NL_HELD  := NL AND REF(LOW, K)  >= REF(MA250, K);
NL_BROKE := NL AND REF(LOW, K)  <  REF(MA250, K);
NH_HELD  := NH AND REF(HIGH, K) <= REF(MA250, K);
NH_BROKE := NH AND REF(HIGH, K) >  REF(MA250, K);

{ ===== 前期高/低 (回踩前 20 根 K 线) ===== }
PRE_HIGH := REF(HHV(HIGH, 20), K + 1);
PRE_LOW  := REF(LLV(LOW,  20), K + 1);

{ ===== 入场信号 (严格按 5_03 讲解) ===== }
LONG_S1 := NL_HELD  AND C > PRE_HIGH AND C > MA250 AND (UPT OR RNG);
LONG_S2 := NL_BROKE AND C > PRE_HIGH AND C > MA250 AND (UPT OR RNG);
BUY_BASE := LONG_S1 OR LONG_S2;

SHORT_S1 := NH_HELD  AND C < PRE_LOW AND C < MA250 AND (DNT OR RNG);
SHORT_S2 := NH_BROKE AND C < PRE_LOW AND C < MA250 AND (DNT OR RNG);
SELL_BASE := SHORT_S1 OR SHORT_S2;

{ ===== 出场 ===== }
X_LONG_WARN  := CROSS(MA120, C);
X_LONG_FULL  := CROSS(MA250, C);
X_SHORT_WARN := CROSS(C, MA120);
X_SHORT_FULL := CROSS(C, MA250);

{ ===== 仓位状态代理 ===== }
MAX_HOLD := 200;
LAST_BUY := BARSLAST(BUY_BASE);
LAST_END_LONG := MIN(BARSLAST(X_LONG_FULL), BARSLAST(SELL_BASE));
IN_LONG := LAST_BUY < LAST_END_LONG AND LAST_BUY <= MAX_HOLD;

LAST_SELL := BARSLAST(SELL_BASE);
LAST_END_SHORT := MIN(BARSLAST(X_SHORT_FULL), BARSLAST(BUY_BASE));
IN_SHORT := LAST_SELL < LAST_END_SHORT AND LAST_SELL <= MAX_HOLD;

EXIT_LONG_WARN  := X_LONG_WARN  AND IN_LONG;
EXIT_LONG_FULL  := X_LONG_FULL  AND IN_LONG;
EXIT_SHORT_WARN := X_SHORT_WARN AND IN_SHORT;
EXIT_SHORT_FULL := X_SHORT_FULL AND IN_SHORT;

{ ===== 价差极端 ===== }
EXT_N := 1200;
SPD := C - MA250;
EXT_L := ABS(SPD) = HHV(ABS(SPD), EXT_N) AND SPD > 0 AND IN_LONG;
EXT_S := ABS(SPD) = HHV(ABS(SPD), EXT_N) AND SPD < 0 AND IN_SHORT;

{ ===== 可视化: STICKLINE 指向 K 线 + DRAWTEXT 标签 ===== }
STICKLINE(BUY_BASE, LOW * 0.997, LOW * 0.989, 2.0, 0), COLORYELLOW;
DRAWTEXT(BUY_BASE, LOW * 0.985, '买'), COLORYELLOW;

STICKLINE(SELL_BASE, HIGH * 1.003, HIGH * 1.011, 2.0, 0), COLORYELLOW;
DRAWTEXT(SELL_BASE, HIGH * 1.015, '卖'), COLORYELLOW;

STICKLINE(EXIT_LONG_FULL, HIGH * 1.002, HIGH * 1.008, 1.5, 0), COLORCYAN;
DRAWTEXT(EXIT_LONG_FULL, HIGH * 1.011, '平多'), COLORCYAN;

STICKLINE(EXIT_SHORT_FULL, LOW * 0.998, LOW * 0.992, 1.5, 0), COLORCYAN;
DRAWTEXT(EXIT_SHORT_FULL, LOW * 0.989, '平空'), COLORCYAN;

STICKLINE(EXIT_LONG_WARN, HIGH * 1.002, HIGH * 1.005, 0.8, 0), COLORWHITE;
DRAWTEXT(EXIT_LONG_WARN, HIGH * 1.008, '减'), COLORWHITE;
STICKLINE(EXIT_SHORT_WARN, LOW * 0.998, LOW * 0.995, 0.8, 0), COLORWHITE;
DRAWTEXT(EXIT_SHORT_WARN, LOW * 0.992, '减'), COLORWHITE;

STICKLINE(EXT_L, HIGH * 1.005, HIGH * 1.013, 1.0, 0), COLORMAGENTA;
DRAWTEXT(EXT_L, HIGH * 1.017, '极'), COLORMAGENTA;
STICKLINE(EXT_S, LOW * 0.995, LOW * 0.987, 1.0, 0), COLORMAGENTA;
DRAWTEXT(EXT_S, LOW * 0.983, '极'), COLORMAGENTA;

{ ===== 趋势染色 ===== }
STICKLINE(UPT AND NOT(BUY_BASE), LOW, LOW * 0.999, 0.5, 0), COLORRED;
STICKLINE(DNT AND NOT(SELL_BASE), HIGH, HIGH * 1.001, 0.5, 0), COLORGREEN;
```

## 图例

每个信号都是**短粗线条 + 文字标签**的组合，线条紧贴对应 K 线，标签在线条另一端——你顺着标签看到的那根粗线，就是信号对应的那根 K 线。

| 标记 | 颜色 | 含义 |
| --- | --- | --- |
| 白线 / 黄线 / 紫线 | 白/黄/紫 | MA20 / MA120 / MA250 |
| K 线下淡红柱 / 上淡绿柱 | 红/绿 | 多头势 / 空头势（MA250 斜率） |
| 粗黄线 + `买` | 黄 | 多头入场（Signal 1 或 Signal 2） |
| 粗黄线 + `卖` | 黄 | 空头入场 |
| 粗青线 + `平多` / `平空` | 青 | MA250 被穿越，全部平仓 |
| 细白线 + `减` | 白 | MA120 被穿越，减仓警告 |
| 紫线 + `极` | 紫 | 价差极端，按方法论全部止盈 |

## 为什么有时候很少入场信号

讲解里作者反复说过：**信号频率不高，需要耐心**（5_02 系统优点第四条："因为是250日均线，所以其实我们这个的周期，其实是不短的"）。

按 5_03 的严格条件：
- 必须有 swing low 出现在 MA250 附近（NL）
- 必须区分理想型/洗盘型
- 必须收盘**突破前期高点**

满足全部条件的机会确实少（讲解里说 5 分钟模型可能"几天才一波"）。如果你在某个品种几个月都看不到信号，可以考虑：

- 把 `NEARP` 从 0.3 调到 0.5 或 1.0（放宽"靠近 MA250"的判定）
- 把 `SLP_E` 调小（让更多 bar 被认定为趋势而非震荡）
- 切换到波动更大的品种

但**不要**通过去掉"突破前期高点"来强行增加信号——那就不是讲解里的系统了。

## 调参

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `SLP_N` | 10 | MA250 斜率回归窗口 |
| `SLP_E` | 0.0001 | 斜率阈值，越大越严格 |
| `K` | 3 | swing 半宽 |
| `NEARP` | 0.3 | 视为靠近 MA250 的百分比距离 |
| `MAX_HOLD` | 200 | 仓位状态最长保持根数（200×5min ≈ 4 个交易日）|
| `EXT_N` | 1200 | 价差极端窗口 |

## 与 Python 版的差异

- **前期高/低窗口**：固定 20 根 K 线，Python 版需要状态机精确找两个 pivot 之间的最高。20 是合理近似。
- **价差排名**：TDX 没有 `rank(pct=True)`，用 `HHV(ABS(spread), N)` 等于当前值近似
- **部分平仓**：只能"信号→提示"
- **多周期对照**：不支持
- **止损"前低 + 1-2 跳"缓冲**：bar 内挂单的事情，主图公式表达不了

## 修订记录

| 版本 | 问题 → 修复 |
| --- | --- |
| v1→v2 | `SLOPE` 与内建函数冲突 → 改 `SLP` |
| v2→v3 | 3 参 `VALUEWHEN` 不是 TDX 原生 → 简化为 "突破 7 根高" |
| v3→v4 | DRAWICON 1/3 显示为 "B"/"空" → 改 DRAWTEXT 明文；加 IN_LONG/IN_SHORT 代理 |
| v4→v5 | 入场太严 + 仓位状态永不出局 → 放宽默认入场、加 MAX_HOLD |
| v5→v6 | 偏离讲解原文 → 回归严格 Signal 1（理想型，回踩守住）+ Signal 2（洗盘型，回踩假突破）+ 共同要求"突破前期高"；DRAWTEXT 改 STICKLINE + DRAWTEXT，线条指向具体 K 线 |
