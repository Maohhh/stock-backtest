# 期货数据集（合并整理版）

本目录是把仓库内分散在多个分支的期货行情数据**合并整理**后的统一版本，全部数据来源于
**新浪财经免费行情接口**，提交进仓库以保证回测**可离线复现**。

- 数据合并自分支：
  - `claude/futures-high-yield-indicator-pozDL` —— 新浪原始行情（主力连续 + 单合约，CSV）
  - `claude/futures-weighted-data-download-LRsDy` —— 持仓量加权连续（Parquet）
- 整理时间：2026-06-16
- 总规模：约 66 MB，776 个数据文件
- 仅供研究回测使用，为公开行情的本地缓存。

---

## 目录结构总览

| 目录 | 内容 | 文件数 | 格式 | 时间范围 |
|---|---|---|---|---|
| `sina_daily_main/` | 日线**主力连续**（后复权拼接） | 43 | CSV | 2005-01-04 ~ 2026-06-08（多数 2013+）|
| `sina_15min_main/` | 15 分钟**主力连续** | 36 | CSV | 各约 1000 根（约 2–3 个月，接口上限）|
| `sina_single_contracts/` | **单个交割月合约**日线（含已退市） | 535 | CSV | 2017-11-20 ~ 2026-06-10 |
| `weighted_15min/` | 15 分钟**持仓量加权连续**（换月不跳空） | 75 | Parquet | 2023-09-04 ~ 2026-06-01 |
| `carry/` | 各品种 carry / 展期结构数据 | 75 | Parquet | 同 `weighted_15min` |
| `manifest_15min.csv` | 加权 15min 全品种清单 | 1 | CSV | — |
| `backtest_results/` | 加权数据上的示例回测产物（非原始数据）| — | csv/png | — |

> 两套数据互补：**sina_*** 是新浪原始行情（主力连续历史长、单合约可重构跨期/跨品种价差）；
> **weighted_15min** 是持仓量加权连续序列（换月无跳空，适合连续回测，但仅近 ~3 年）。

---

## 各数据集字段

### `sina_daily_main/`（日线主力连续，如 `RB0.csv`）
列：`d, o, h, l, c, v, p, s`
（日期、开、高、低、收、成交量、持仓量、结算价）。文件名后缀 `0` 表示连续主力。

### `sina_15min_main/`（15 分钟主力连续，如 `RB0.csv`）
列：`d, o, h, l, c, v, p`（时间、开、高、低、收、量、持仓量）。
受新浪接口限制每合约约 1000 根 K 线，覆盖最近约 2–3 个月。

### `sina_single_contracts/`（单合约日线，如 `RB2510.csv`）
列：`d, c`（日期、收盘价）。文件名为「品种+交割年月」，含已退市合约，
主要用于按同月合约**重构跨品种 / 跨期价差**（玉米-淀粉、PVC-聚丙烯、棉花-棉纱等）。

### `weighted_15min/`（持仓量加权连续，如 `RB.parquet`）
列：`datetime, open, high, low, close, volume, open_interest, n_contracts`。
同品种所有在交易合约按各自持仓量在每根 K 线上加权合成，换月不跳空。
`n_contracts` 为参与加权的合约数（越靠近当前越完整，最早几根可能为 1）。

### `carry/`（展期结构，如 `RB.parquet`）
各品种近远月 carry / 展期收益结构数据，配合加权连续做跨期与 carry 类策略。

### `manifest_15min.csv`
加权 15min 全品种清单，列：`product, exchange, name, rows, start, end, max_contracts, complete`。

---

## 覆盖品种

**日线主力连续 / 加权 15min（主要品种）**：
黑色(RB/HC/I/J/JM/SS/SF/SM)、有色(CU/AL/ZN/NI/SN/PB/AO/BC)、
能化(TA/MA/PP/L/V/EG/FU/BU/SC/RU/PG/EB/PX/SA/UR/FG)、
农产品(M/Y/P/C/CS/A/B/RM/OI/CF/SR/AP/JD/LH/RR/CJ)、
贵金属(AU/AG)、新能源(LC/SI/PS/BR)、股指(IF/IH/IC/IM)、
国债(T/TF/TL/TS)、广期所(SH/LG) 等。

`weighted_15min/` 覆盖 75 个品种（上期所/能源中心/大商所/郑商所/中金所/广期所主要品种）；
`sina_daily_main/` 覆盖 43 个；`sina_15min_main/` 覆盖 36 个。

**单合约（27 个品种的主力月份）**：
A B C CF CS CY FG HC I J JM L M MA OI P PF PP PR PX RB RM SA SF SM TA V Y

---

## 读取示例

```python
import pandas as pd

# 日线主力连续
rb_daily = pd.read_csv("data_futures/sina_daily_main/RB0.csv")

# 加权连续 15min
rb_w = pd.read_parquet("data_futures/weighted_15min/RB.parquet")

# 单合约（重构价差）
c2509 = pd.read_csv("data_futures/sina_single_contracts/C2509.csv")

# 全品种清单
manifest = pd.read_csv("data_futures/manifest_15min.csv")
```

---

## 复现 / 更新

原始抓取脚本位于对应特性分支（未随数据合并进本分支）：
- 新浪原始行情：`src/data/futures.py`、`src/data/contracts.py`
  （见分支 `claude/futures-high-yield-indicator-pozDL`）
- 加权连续下载器：`download_futures.py`
  （见分支 `claude/futures-weighted-data-download-LRsDy`）

数据为公开行情的本地缓存；新浪分钟线接口对高频请求会返回 HTTP 456 封禁，
脚本已做限速与指数退避重试。如需更长/更精确历史，请改用付费源（天勤 TqSdk、Tushare Pro、Wind、米筐等）。
