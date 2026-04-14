# 数据获取模块

import pandas as pd
from typing import Optional, Literal
from datetime import datetime
import requests
import json


class SinaDataSource:
    """Sina 财经数据源"""
    
    BASE_URL = "https://quotes.sina.cn/cn/api/quotes.php"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
    
    def get_daily_data(
        self, 
        symbol: str, 
        start: str, 
        end: str
    ) -> Optional[pd.DataFrame]:
        """
        获取日线数据
        
        Args:
            symbol: 股票代码，如 "000001.SZ"
            start: 开始日期，格式 "YYYY-MM-DD"
            end: 结束日期，格式 "YYYY-MM-DD"
            
        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        try:
            # 使用 Sina 历史数据接口
            # symbol 转换: 000001.SZ -> sz000001, 000001.SH -> sh000001
            sina_symbol = self._to_sina_symbol(symbol)
            url = f"https://quotes.sina.cn/cn/api/quotes.php?symbol={sina_symbol}&source=chart"
            
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            # Sina 返回的是 JavaScript 变量赋值格式，提取 JSON 部分
            text = response.text
            # 尝试直接解析为 JSON，或从 js 变量中提取
            if text.startswith("var"):
                json_start = text.find("{")
                json_end = text.rfind("}") + 1
                if json_start == -1 or json_end == 0:
                    return None
                data = json.loads(text[json_start:json_end])
            else:
                data = response.json()
            
            # Sina 日线数据通常在 data 或 result 中，格式为列表
            # 不同接口返回结构可能不同，这里做兼容处理
            records = data.get("data") or data.get("result") or data.get("day") or []
            if not records:
                return None
            
            # 统一解析为 DataFrame
            rows = []
            for record in records:
                if isinstance(record, list) and len(record) >= 6:
                    # [date, open, high, low, close, volume]
                    rows.append({
                        "date": record[0],
                        "open": float(record[1]),
                        "high": float(record[2]),
                        "low": float(record[3]),
                        "close": float(record[4]),
                        "volume": float(record[5]),
                    })
                elif isinstance(record, dict):
                    rows.append({
                        "date": record.get("date") or record.get("day"),
                        "open": float(record.get("open", 0)),
                        "high": float(record.get("high", 0)),
                        "low": float(record.get("low", 0)),
                        "close": float(record.get("close", 0)),
                        "volume": float(record.get("volume", 0)),
                    })
            
            if not rows:
                return None
            
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            
            # 按日期过滤
            df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
            
            if df.empty:
                return None
            
            return df[["date", "open", "high", "low", "close", "volume"]]
            
        except Exception as e:
            print(f"Sina 数据源获取失败: {e}")
            return None
    
    def _to_sina_symbol(self, symbol: str) -> str:
        """将标准股票代码转换为 Sina 格式"""
        symbol = symbol.strip()
        if "." in symbol:
            code, exchange = symbol.split(".")
            exchange = exchange.lower()
            if exchange == "sz":
                return f"sz{code}"
            elif exchange == "sh":
                return f"sh{code}"
            elif exchange == "bj":
                return f"bj{code}"
        return symbol
    
    def get_realtime_quote(self, symbol: str) -> Optional[dict]:
        """获取实时行情"""
        try:
            sina_symbol = self._to_sina_symbol(symbol)
            # Sina 实时行情接口
            url = f"https://hq.sinajs.cn/list={sina_symbol}"
            response = self.session.get(url, timeout=10)
            response.encoding = "gb2312"
            
            text = response.text.strip()
            # 格式: var hq_str_sz000001="..."
            if "hq_str_" not in text:
                return None
            
            parts = text.split('="')
            if len(parts) < 2:
                return None
            
            data_str = parts[1].rstrip('";')
            fields = data_str.split(",")
            
            if len(fields) < 5:
                return None
            
            return {
                "symbol": symbol,
                "name": fields[0] if len(fields) > 0 else "",
                "open": float(fields[1]) if len(fields) > 1 else None,
                "close": float(fields[2]) if len(fields) > 2 else None,
                "current": float(fields[3]) if len(fields) > 3 else None,
                "high": float(fields[4]) if len(fields) > 4 else None,
                "low": float(fields[5]) if len(fields) > 5 else None,
            }
        except Exception as e:
            print(f"Sina 实时行情获取失败: {e}")
            return None


class AkShareDataSource:
    """AkShare 开源财经数据源"""
    
    def __init__(self):
        try:
            import akshare as ak
            self.ak = ak
        except ImportError:
            raise ImportError("请先安装 akshare: pip install akshare")
    
    def get_daily_data(
        self, 
        symbol: str, 
        start: str, 
        end: str
    ) -> Optional[pd.DataFrame]:
        """
        获取日线数据
        
        Args:
            symbol: 股票代码，如 "000001"
            start: 开始日期，格式 "YYYYMMDD"
            end: 结束日期，格式 "YYYYMMDD"
            
        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        try:
            # 转换日期格式
            start_fmt = start.replace("-", "")
            end_fmt = end.replace("-", "")
            
            # 使用 akshare 获取数据
            df = self.ak.stock_zh_a_hist(
                symbol=symbol.split(".")[0],
                start_date=start_fmt,
                end_date=end_fmt,
                adjust="qfq"  # 前复权
            )
            
            if df is None or df.empty:
                return None
            
            # 标准化列名
            df = df.rename(columns={
                "日期": "date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume"
            })
            
            return df[["date", "open", "high", "low", "close", "volume"]]
            
        except Exception as e:
            print(f"AkShare 数据源获取失败: {e}")
            return None


class DataManager:
    """
    数据管理器
    
    自动管理多个数据源，支持 fallback 机制
    """
    
    def __init__(self):
        self.sina = SinaDataSource()
        self.akshare = AkShareDataSource()
    
    def get_daily_data(
        self, 
        symbol: str, 
        start: str, 
        end: str,
        source: Literal["auto", "sina", "akshare"] = "auto"
    ) -> pd.DataFrame:
        """
        获取日线数据，支持自动 fallback
        
        Args:
            symbol: 股票代码
            start: 开始日期
            end: 结束日期
            source: 数据源选择，默认 auto 自动选择
            
        Returns:
            DataFrame
            
        Raises:
            ValueError: 所有数据源都失败时抛出
        """
        if source == "sina":
            df = self.sina.get_daily_data(symbol, start, end)
            if df is not None:
                return df
            raise ValueError(f"Sina 数据源获取 {symbol} 数据失败")
        
        elif source == "akshare":
            df = self.akshare.get_daily_data(symbol, start, end)
            if df is not None:
                return df
            raise ValueError(f"AkShare 数据源获取 {symbol} 数据失败")
        
        else:  # auto
            # 先尝试 Sina
            df = self.sina.get_daily_data(symbol, start, end)
            if df is not None:
                print(f"✅ 使用 Sina 数据源获取 {symbol} 数据")
                return df
            
            # Fallback 到 AkShare
            print(f"⚠️ Sina 失败，尝试 AkShare 数据源...")
            df = self.akshare.get_daily_data(symbol, start, end)
            if df is not None:
                print(f"✅ 使用 AkShare 数据源获取 {symbol} 数据")
                return df
            
            raise ValueError(f"所有数据源获取 {symbol} 数据均失败")
    
    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        try:
            return self.akshare.ak.stock_zh_a_spot_em()
        except Exception as e:
            print(f"获取股票列表失败: {e}")
            return pd.DataFrame()