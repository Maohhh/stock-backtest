# 期货求魔系统 — 通达信公式

本目录把 `src/indicators/qiumo.py` 的核心方法论用通达信公式语言重写。完整的算法说明见 [`../METHOD.md`](../METHOD.md)。

## 文件

| 文件 | 类型 | 用途 |
| --- | --- | --- |
| `qiumo_main.tdx` | 主图叠加 | 在 K 线图上画 MA20/120/250、趋势染色、入场箭头、出场叉号、价差极端文字 |
| `qiumo_xuangu.tdx` | 条件选股 | 扫描当前周期满足"多头入场"的标的列表（做空版自己镜像即可）|

## 怎么导入到通达信

1. 打开通达信终端 → 顶部菜单 **工具** → **公式管理器**（也可用快捷键 `Ctrl+F`）。
2. 主图：左侧树选 **技术指标** → **其他类型**（或你常用的分类）→ 点击 **新建**，名称填 `QIUMO`，类型选 **主图叠加**。把 `qiumo_main.tdx` 内容粘到右侧编辑框，点 **测试公式** 确认无报错，点 **确定** 保存。
3. 选股：左侧树选 **条件选股公式** → 新建 → 名称 `QIUMO_LONG`，把 `qiumo_xuangu.tdx` 内容粘进去，保存。
4. 用法：在 K 线图右键 → **加载指标** → 选 `QIUMO`；或菜单 **功能** → **选股器** → **条件选股** → 选 `QIUMO_LONG`。

## 图上你会看到什么

- **白线** = MA20（快/出场参考）
- **黄线** = MA120（中/部分止盈参考）
- **紫线** = MA250（慢/趋势锚，最后底线）
- **K 线下方淡红柱** = 多头势（MA250 斜率向上）
- **K 线上方淡绿柱** = 空头势（MA250 斜率向下）
- **上箭头**（K 线下方）= 多头入场信号
- **下箭头**（K 线上方）= 空头入场信号
- **叉号**（K 线上/下方）= 全平信号（收盘跌破 MA250 / 站上 MA250）
- **小圆点** = 减仓警告（收盘跌破 MA120 / 站上 MA120），按方法论应平大部分仓位留小底仓——通达信无法直接表达部分平仓，请手动操作
- **黄色文字"极端拉升/杀跌 全部止盈"** = 价差扩张到近 1200 根（约 1 周）最大值，按方法论该全部止盈

## 参数与调参

公式开头几个 `:=` 是常用调整点：

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `SLOPE_N` | 10 | MA250 斜率回归窗口（根 K 线）|
| `SLOPE_E` | 0.0001 | 斜率阈值；调大 → 更少触发趋势 |
| `K` | 3 | swing pivot 半宽；商品波动大改 5 |
| `NEAR_PCT` | 0.3 | 视为"靠近 MA250"的百分比距离 |
| `EXT_N` | 1200 | 价差极端窗口（5min 约 1 周；改 12000 ≈ 1 年）|

## 与 Python 版的差异（已知简化）

| 差异 | Python 版 | 通达信版 |
| --- | --- | --- |
| 价差排名 | rolling percentile rank | `HHV(ABS(spread), N)` 等于当前值才认极端，更稀疏 |
| Signal1 W 底 | 真状态机检测中间高突破 | 简化为"靠近底 + 高低相等以上 + 突破最近 7 根高" |
| 部分平仓 | 支持 `use_partial_exit` | 不支持，只给警告 ICON |
| 多周期对照 | 调用方实现 | 不支持，本公式只看当前周期 |
| 止损"前低 + 1-2 跳"缓冲 | 事件驱动可加 | 不支持 |

## 把做空选股也搞出来

`qiumo_xuangu.tdx` 只输出做多。做空对称：

```
{ 做空选股版 - 自己保存为 QIUMO_SHORT }
SLOPE_N := 10;
K := 3;
NEAR_PCT := 0.3;

MA250 := MA(C, 250);
SLOPE := (MA250 - REF(MA250, SLOPE_N)) / C;
DNT := SLOPE < -0.0001;
RNG := NOT(DNT) AND NOT(SLOPE > 0.0001);

PH := REF(HIGH, K) = HHV(HIGH, 2*K+1);
NH := PH AND ABS(REF(HIGH, K) - REF(MA250, K)) / REF(MA250, K) * 100 <= NEAR_PCT;

LAST_NH := VALUEWHEN(1, NH, REF(HIGH, K));
PREV_NH := VALUEWHEN(2, NH, REF(HIGH, K));

SHORT_S2 := NH AND LAST_NH < PREV_NH AND C < MA250 AND (DNT OR RNG);
SHORT_S1 := NH AND LAST_NH <= PREV_NH AND C < LLV(LOW, 7) AND (DNT OR RNG);

CC: SHORT_S1 OR SHORT_S2;
```

## 兼容性

- 公式只用了 `MA / REF / HHV / LLV / VALUEWHEN / CROSS / DRAWICON / DRAWTEXT / STICKLINE / ABS` 这些**通达信标准函数**，应该在通达信金融终端、同花顺（部分函数兼容）、东方财富通也能运行；颜色常量按通达信写。
- 若用同花顺要注意 `DRAWICON` 编号体系略有差异，按平台 ICON 表替换即可。
