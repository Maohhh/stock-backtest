# 期货全品种 15 分钟加权数据

本目录存放期货全品种的 **加权(指数)** 15 分钟 K 线数据，由仓库根目录的
`download_futures_15m.py` 下载生成。

## 数据来源
- 东方财富历史 K 线接口（`push2his.eastmoney.com`，`klt=15`，前复权 `fqt=1`）。
- 加权/指数序列位于东财市场号 159，代码形如 `159.rbfi`（螺纹钢加权）、`159.cufi`（沪铜加权）。
- 品种 -> 加权代码映射见 `src/data/futures_weighted_symbols.json`（共 70 个商品期货品种）。

## 文件格式
- 每个品种一个文件，文件名为加权代码（如 `rbfi.parquet` / `cufi.parquet`）。
- 列：`datetime, open, high, low, close, volume, amount`。

## 如何生成
```bash
pip install -r requirements.txt
python download_futures_15m.py                 # 全品种近 5 年
python download_futures_15m.py --products 螺纹钢,沪铜   # 指定品种
```

## 注意
- 历史 K 线主机 `push2his.eastmoney.com` 在部分受限网络/云环境会返回 503
  （upstream connect error）。脚本启动时会做连通性预检；如不通，请在可访问该
  主机的网络环境（本地机器或放开出网策略的环境）中运行。
- 金融期货（股指 IF/IH/IC/IM、国债 TS/TF/T/TL）在东财无 159 加权指数，未纳入。
