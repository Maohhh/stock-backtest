# 期货全品种 15 分钟加权数据

用于回测的国内期货**加权连续**(open-interest weighted continuous) 15 分钟 K 线数据。

## 什么是「加权」连续

单个期货品种(如螺纹钢 RB)同一时间有多个在交易的月份合约(RB2510、RB2601、RB2605…)。
「加权」连续序列把同品种**所有在交易合约**按各自的**持仓量(open interest)** 在每一根
K 线上加权合成一条价格序列：

```
close_weighted(t) = Σ_i  close_i(t) · oi_i(t)  /  Σ_i oi_i(t)
```

open/high/low 同样按持仓量加权，volume 与 open_interest 为各合约求和。

与「主力连续」相比，加权序列在换月时**不会跳空**(主力换月会产生缺口)，更适合连续回测、
指标计算和跨期比较。

## 数据来源与限制

- **数据源**：新浪财经免费分钟线接口 `getFewMinLine`。
- **历史深度**：该接口对每个合约**最多只返回最近 1023 根** 15 分钟 K 线(约 2 个月)。
  为突破此限制，下载器会**额外抓取已到期的历史合约**(`--back-months`)——每个合约都返回
  它到期前最后 1023 根，按到期月份依次拼接，即可把加权序列向前扩展到 **1 年以上**
  (本仓库当前数据约覆盖近 13 个月: 2025-04 ~ 2026-06)。
- **较老时段的加权口径**：由于每个合约只有到期前 ~2.5 个月数据，越往历史回溯，能覆盖某个
  时间点的合约越少(只剩临近到期的近月合约)，因此**较老时段是近月持仓量加权**而非全合约
  加权。好在远月持仓量本来就小、对加权贡献低，近月加权是全加权的良好近似，且换月连续无跳空。
  最近 ~2.5 个月则是全合约真加权(`n_contracts` 可达 10+)。
- **如需更长/更精确历史**：请改用付费/专业源(天勤 TqSdk、Tushare Pro、Wind、米筐等)。
- **反爬**：接口对高频请求返回 HTTP 456 封禁。下载脚本已做顺序抓取 + 限速 +
  指数退避重试，被封禁时会自动冷却后续抓，保证最终数据完整。
- **覆盖范围**：上期所/能源中心/大商所/郑商所/中金所/广期所全部主要品种。
  已停牌或长期无在交易合约的品种(如普麦 PM、早籼稻 RI、粳稻 JR、晚籼稻 LR、棉纱 CY 等)
  会被自动跳过。

## 目录结构

```
data_futures/
├── 15min/
│   ├── RB.parquet      # 螺纹钢加权 15min
│   ├── M.parquet       # 豆粕加权 15min
│   └── ...             # 每个品种一个文件
└── manifest_15min.csv  # 清单: 品种/交易所/中文名/行数/起止时间/最大合约数
```

## 数据字段(每个 parquet)

| 列 | 含义 |
|---|---|
| `datetime` | K 线时间(每根的结束时间，含夜盘) |
| `open` / `high` / `low` / `close` | 持仓量加权后的开/高/低/收 |
| `volume` | 各在交易合约成交量之和 |
| `open_interest` | 各在交易合约持仓量之和 |
| `n_contracts` | 该根 K 线参与加权的合约数 |

> 说明：序列最早的少数几根 K 线 `n_contracts` 可能为 1，因为远月合约的 1023 根
> 历史窗口回溯不到那么早；越靠近当前，参与加权的合约越完整。

## 读取示例

```python
import pandas as pd

rb = pd.read_parquet("data_futures/15min/RB.parquet")
print(rb.tail())

# 全品种清单
manifest = pd.read_csv("data_futures/manifest_15min.csv")
print(manifest)
```

## 重新下载 / 更新

```bash
# 全品种 15 分钟加权(默认)
python download_futures.py

# 指定品种
python download_futures.py --products RB,M,CU,IF

# 换周期(1/5/15/30/60 分钟)
python download_futures.py --period 5

# 扩展历史: 向前回溯 14 个月的到期合约, 保留近 410 天
python download_futures.py --back-months 14 --window-days 410

# 调节抓取节奏(被反爬封禁时可调大 --rate)
python download_futures.py --rate 0.5 --cooldown 60
```

依赖：`pip install pandas pyarrow requests`

## 回测：期货求魔·趋势过滤版 V2

`backtest_qiumo.py` 把同花顺主图指标「期货求魔·趋势过滤版 V2」逐行翻译为 Python，
并在上述 15 分钟加权数据上做事件驱动回测。

- **核心逻辑**：MA250 附近的 pivot 回踩 + 突破前高/前低入场，叠加 `MA(C,1500)` 宏观趋势门
  (只在 C>MA1500 做多、C<MA1500 做空)。
- **三档分批离场**：极平(极端价差)减 1/3、减仓(破 MA120)减 1/3、平多/平空(破 MA250)清剩余；
  另有止损(`STOP_L/S`)、反向信号、超时(200 根)全平。
- **成本**：默认每边 2bp(滑点+手续)。

```bash
python backtest_qiumo.py                 # 全品种
python backtest_qiumo.py --products AG,CU,BU
```

结果写入 `data_futures/backtest_qiumo/`：`by_product.csv`(分品种绩效) 与 `trades.csv`(逐笔)。

**当前数据(近13个月)回测概况**：71 个品种、1742 笔交易，整体接近盈亏平衡
(胜率 ~27%、PF ~1.0)，但品种分化极大——贵金属/有色/能化趋势品种(AG +43%、CU +20%、
BU +22%、FU +18%)表现好，螺纹/镍/豆粕等(WR、NI、M)亏损。属典型低胜率趋势策略，
需配合品种筛选与组合分散使用。

> 注：`EXT_N=12000` 极平回溯窗口在 15min 数据上长于现有 K 线数，脚本按同花顺习惯用
> 现有全部 K 线取极值；该参数对极平触发频率较敏感，换周期时请按指标注释同比调整。
