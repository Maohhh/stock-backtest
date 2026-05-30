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
- **历史深度**：该接口对每个合约**最多只返回最近 1023 根** 15 分钟 K 线(约 2 个月)，
  因此加权序列历史深度同样约 **~2 个月**。这是免费数据源的硬限制，无法绕过。
  如需更长历史，请改用付费/专业源(天勤 TqSdk、Tushare Pro、Wind、米筐等)。
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

# 调节抓取节奏(被反爬封禁时可调大 --rate)
python download_futures.py --rate 0.5 --cooldown 60
```

依赖：`pip install pandas pyarrow requests`
