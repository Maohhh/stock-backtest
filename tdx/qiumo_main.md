# 期货求魔系统 — 通达信主图叠加指标

来源：bilibili UP「期货求魔」9 集教学。完整方法论见 [`METHOD.md`](../METHOD.md)，Python 实现见 [`src/indicators/qiumo.py`](../src/indicators/qiumo.py)。

## 导入步骤

1. 通达信菜单 **工具** → **公式管理器**（快捷键 `Ctrl+F`）
2. 左侧选 **技术指标** → 你的分类目录 → **新建**
3. 公式名称：`QIUMO`，类型选 **主图叠加**
4. 把下方公式整段粘进右侧编辑框 → **测试公式** → **确定**
5. 回到 K 线图右键 **加载指标** 选 `QIUMO`

## 公式（v4，DRAWTEXT 明文 + 仓位状态过滤）

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

{ ===== 入场基本信号 ===== }
BUY_BASE  := NL AND C > MA250 AND C > HHV(HIGH, 7) AND (UPT OR RNG);
SELL_BASE := NH AND C < MA250 AND C < LLV(LOW, 7) AND (DNT OR RNG);

{ ===== 出场基本事件 ===== }
X_LONG_WARN  := CROSS(MA120, C);
X_LONG_FULL  := CROSS(MA250, C);
X_SHORT_WARN := CROSS(C, MA120);
X_SHORT_FULL := CROSS(C, MA250);

{ ===== 仓位状态代理 (粗略, 但够过滤噪音) ===== }
IN_LONG  := BARSLAST(BUY_BASE)  < BARSLAST(X_LONG_FULL);
IN_SHORT := BARSLAST(SELL_BASE) < BARSLAST(X_SHORT_FULL);

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

{ ===== 主图标记 (DRAWTEXT 明文) ===== }
DRAWTEXT(BUY_BASE,         LOW  * 0.998, '买'),   COLORRED;
DRAWTEXT(SELL_BASE,        HIGH * 1.002, '卖'),   COLORGREEN;
DRAWTEXT(EXIT_LONG_FULL,   HIGH * 1.002, '平多'), COLORYELLOW;
DRAWTEXT(EXIT_SHORT_FULL,  LOW  * 0.998, '平空'), COLORYELLOW;
DRAWTEXT(EXIT_LONG_WARN,   HIGH * 1.002, '减'),   COLORWHITE;
DRAWTEXT(EXIT_SHORT_WARN,  LOW  * 0.998, '减'),   COLORWHITE;
DRAWTEXT(EXT_L,            HIGH * 1.005, '极'),   COLORYELLOW;
DRAWTEXT(EXT_S,            LOW  * 0.995, '极'),   COLORYELLOW;

{ ===== 趋势染色 ===== }
STICKLINE(UPT, LOW,  LOW  * 0.998, 1.5, 0), COLORRED;
STICKLINE(DNT, HIGH, HIGH * 1.002, 1.5, 0), COLORGREEN;
```

## 图例

| 标记 | 颜色 | 含义 |
| --- | --- | --- |
| 白线 | 白 | MA20（快/出场参考） |
| 黄线 | 黄 | MA120（中/部分止盈参考） |
| 紫线（粗） | 紫 | MA250（慢/趋势锚） |
| K 线下淡红柱 | 红 | 多头势（MA250 斜率向上） |
| K 线上淡绿柱 | 绿 | 空头势（MA250 斜率向下） |
| `买` | 红 | 多头入场 |
| `卖` | 绿 | 空头入场 |
| `平多` | 黄 | 多头全平（MA250 跌破，估计在多仓时才显示） |
| `平空` | 黄 | 空头全平（MA250 站上，估计在空仓时才显示） |
| `减` | 白 | 减仓警告（MA120 被穿，按方法论该减仓） |
| `极` | 黄 | 价差极端（按方法论该全部止盈） |

## 仓位状态代理（v4 重点）

通达信公式无法真正跟踪你的实盘仓位，但用 `BARSLAST` 比较"最近一次入场"和"最近一次全平"哪个更新，可以做一个**粗略代理**：

```
IN_LONG := BARSLAST(BUY_BASE) < BARSLAST(X_LONG_FULL);
```

意思是"最近一次买入信号比最近一次 MA250 跌破信号更新"→ 估计仍在多仓。所有"平多/减"标记都过这一关，避免震荡市里 MA250 来回穿满屏标"平"。

**局限**：
- 假设了每次 `BUY_BASE` = 满仓，每次 `MA250` 跌破 = 全平。不能反映你实际的部分加减仓。
- 实盘可能你并未按信号建仓；那"IN_LONG"对你就只是"系统认为应该持多"的状态，仅供参考。

## 调参

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `SLP_N` | 10 | MA250 斜率回归窗口 |
| `SLP_E` | 0.0001 | 斜率阈值，越大越严格 |
| `K` | 3 | swing 半宽（波动大的品种调到 5） |
| `NEARP` | 0.3 | 视为靠近 MA250 的百分比距离 |
| `EXT_N` | 1200 | 价差极端窗口（5min ≈ 1 周；12000 ≈ 1 年） |

## 与 Python 版的差异（已知简化）

- **价差排名**：TDX 没有 `rank(pct=True)`，用 `HHV(ABS(spread), N)` 等于当前值近似，只在打破新高极端才触发，比 Python 稀疏
- **入场信号合并**：Python 版区分 W 底 / 抬高底两个信号；通达信合成一个"pivot + 突破"条件，等效但稍宽松
- **部分平仓**：只能"信号→提示"，不能真正控仓位
- **仓位状态**：v4 的 IN_LONG/IN_SHORT 是粗略代理，与实盘可能不一致
- **多周期对照**：本公式只看当前周期
- **止损"前低 + 1-2 跳"缓冲**：bar 内挂单的事情，主图公式表达不了

## 踩过的兼容性坑（修订记录）

| 版本 | 问题 → 修复 |
| --- | --- |
| v1→v2 | `SLOPE` 与通达信内建函数同名 → 改 `SLP`；`OPEN_*` 防与 OPEN 关键字组合 → 改 `*_SIG`；`SPREAD` → `SPD` |
| v2→v3 | `VALUEWHEN(N, COND, X)` 三参数是大智慧/同花顺扩展，通达信原生只支持 `VALUEWHEN(COND, X)` 两参数 → 去掉历史 pivot 比较，改用"突破 7 根高"作为等效抬高底确认 |
| v3→v4 | DRAWICON 1/3 在通达信渲染为 "B"/"空" 文字（用户以为是 bug 实际是 ICON 编号），且 MA250 上下穿在震荡市过于频繁 → 改用 DRAWTEXT 明文 + 加 BARSLAST 仓位状态代理过滤 |
