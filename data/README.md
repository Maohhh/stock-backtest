# 期货数据集（爬取自新浪财经）

本目录为回测所用的全部原始行情数据，由 `src/data/futures.py` 与 `src/data/contracts.py`
从新浪财经接口抓取并缓存。提交进仓库以保证回测**可离线复现**、不依赖重新下载。

数据列：`d/o/h/l/c/v`（时间、开、高、低、收、量）；连续主力为后复权拼接。

## 目录结构

| 目录 | 内容 | 文件数 | 说明 |
|---|---|---|---|
| `futures_15min/` | 37 个品种 15 分钟K线 | ~36 | 各约 1000 根（新浪接口上限，约 2–3 个月） |
| `futures_daily/` | 43 个品种日线连续主力 | 43 | 最长回溯至 2005，多数 2013+ |
| `futures_contracts/` | 单合约（具体交割月）日线 | ~482 | 含已退市合约，约 2018 至今；用于重构跨品种/跨期价差 |

## 命名

- 连续主力：`RB0.csv`、`C0.csv`（后缀 0）
- 单合约：`RB2510.csv`、`C2509.csv`（品种+年月）

## 覆盖品种

- 15min / 日线：黑色(RB/HC/I/J/JM/SS/SF/SM)、有色(CU/AL/ZN/NI/SN/PB)、
  能化(TA/MA/PP/L/V/EG/FU/BU/SC/RU)、农产品(M/Y/P/C/CS/A/RM/OI/CF/SR/AP/JD/B)、
  贵金属(AU/AG)、股指(IF/IH/IC/IM)、国债(T/TF) 等。
- 单合约：C/CS/RB/HC/I/J/JM/M/RM/P/Y/OI/A/B/V/PP/L/CF/CY/SF/SM 的主力月份。

## 用途对照

- 横截面反转 / 趋势指标 → `futures_daily/`、`futures_15min/`
- 跨品种套利篮子（玉米-淀粉 / PVC-聚丙烯 / 棉花-棉纱）→ `futures_contracts/`（同月价差重构）

## 复现

删除本目录后，运行任一 `run_*.py` 脚本会自动重新下载（需联网）。
数据为公开行情的本地缓存，仅供研究回测使用。
