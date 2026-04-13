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
            # Sina API 实现
            # TODO: 实现具体的 API 调用
            pass
        except Exception as e:
            print(f"Sina 数据源获取失败: {e}")
            return None
    
    def get_realtime_quote(self, symbol: str) -> Optional[dict]:
        """获取实时行情"""
        try:
            # TODO: 实现实时行情获取
            pass
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