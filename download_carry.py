"""为 carry/期限结构因子构建分合约数据。

复用 download_futures 的抓取逻辑, 对每个品种抓取其历史所有月份合约,
对每个时间点取持仓量最高的两个合约(近月/次月), 计算年化展期收益(carry):
    carry = (P_near - P_far) / P_far / (Δmonth/12)        (P_near 近月, P_far 远月)
backwardation(near>far)=正carry=现货溢价; contango=负carry。
输出每品种的日频 carry 序列 -> data_futures/carry/{prod}.parquet
"""
import glob
import re
import numpy as np
import pandas as pd
import requests

import download_futures as dl

WINDOW_DAYS = 1000
OUT = "data_futures/carry"


def month_num(ym):
    y, m = int(ym[:2]), int(ym[2:])
    return (2000 + y) * 12 + m


def build_carry(product, months, session, limiter):
    # 抓取该品种所有候选合约的 (datetime, close, oi, expiry_month)
    frames = []
    seen_real = False
    empty = 0
    for ym in months:
        contract = f"{product}{ym}"
        try:
            df = dl.fetch_contract(session, contract, 15, limiter)
        except dl.Throttled:
            continue
        if df is not None and not df.empty:
            df = df[["datetime", "close", "open_interest"]].copy()
            df["exp"] = month_num(ym)
            frames.append(df)
            seen_real = True
            empty = 0
        elif seen_real:
            empty += 1
            if empty >= 99:   # 长历史不早停
                pass
    if not frames:
        return None
    allc = pd.concat(frames, ignore_index=True)
    allc = allc[allc["open_interest"] > 0]
    if allc.empty:
        return None
    # 每个时间点: 按OI排序取前两个合约, 近月=exp较小者
    rows = []
    for dt, g in allc.groupby("datetime"):
        if g["exp"].nunique() < 2:
            continue
        top2 = g.sort_values("open_interest", ascending=False).head(2)
        top2 = top2.sort_values("exp")
        near, far = top2.iloc[0], top2.iloc[1]
        dm = far["exp"] - near["exp"]
        if dm <= 0 or near["close"] <= 0 or far["close"] <= 0:
            continue
        carry = (near["close"] - far["close"]) / far["close"] / (dm / 12.0)
        rows.append({"datetime": dt, "carry": carry,
                     "near": near["close"], "far": far["close"], "dm": dm})
    if not rows:
        return None
    out = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
    # 日频(取每日最后)
    out["date"] = pd.to_datetime(out["datetime"]).dt.normalize()
    daily = out.groupby("date").last().reset_index()
    return daily


def main():
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    os.makedirs(OUT, exist_ok=True)
    months = dl.candidate_months(36, 15)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; carry)"})
    limiter = dl.RateLimiter(0.25)        # 全局限速(所有线程共享)
    # 跳过已完成的品种(断点续传)
    done = {os.path.basename(f)[:-8] for f in glob.glob(f"{OUT}/*.parquet")}
    products = [p for p in dl.PRODUCTS if p not in done]
    print(f"待处理 {len(products)} 品种(已完成 {len(done)})", flush=True)

    def work(product):
        try:
            d = build_carry(product, months, session, limiter)
        except Exception as e:
            return product, None, str(e)
        return product, d, None

    ok = len(done)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(work, p): p for p in products}
        for fut in as_completed(futs):
            product, d, err = fut.result()
            if err:
                print(f"⚠️ {product} 异常 {err}", flush=True)
            elif d is None or len(d) < 50:
                print(f"⚠️ {product} carry数据不足", flush=True)
            else:
                d.to_parquet(f"{OUT}/{product}.parquet", index=False)
                ok += 1
                print(f"✅ {product} {len(d)}日 carry均值={d['carry'].mean():+.3f} "
                      f"({d['date'].min().date()}~{d['date'].max().date()})", flush=True)
    print(f"完成: {ok} 品种")


if __name__ == "__main__":
    main()
