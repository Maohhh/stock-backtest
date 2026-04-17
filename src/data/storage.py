"""
数据持久化模块

支持 SQLite 和 Parquet 两种存储格式
"""

import pandas as pd
import numpy as np
from typing import Optional, List
from datetime import datetime
from pathlib import Path
import json


class SQLiteStorage:
    """SQLite 数据持久化"""
    
    def __init__(self, db_path: str = "data/stock_data.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def save_daily_data(self, symbol: str, df: pd.DataFrame) -> None:
        """保存日线数据"""
        import sqlite3
        
        conn = sqlite3.connect(str(self.db_path))
        try:
            # 添加元信息
            df_to_save = df.copy()
            df_to_save['symbol'] = symbol
            df_to_save['updated_at'] = datetime.now().isoformat()
            
            # 使用 REPLACE 策略，按 symbol + date 去重
            df_to_save.to_sql('daily_data', conn, if_exists='append', index=False)
            
            # 创建去重视图
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_data_dedup AS
                SELECT * FROM daily_data WHERE 1=0
            """)
            
            # 清理重复数据（保留最新）
            conn.execute("""
                DELETE FROM daily_data
                WHERE rowid NOT IN (
                    SELECT MIN(rowid)
                    FROM daily_data
                    GROUP BY symbol, date
                )
            """)
            
            # 创建索引
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_daily_data_symbol_date
                ON daily_data(symbol, date)
            """)
            
            conn.commit()
        finally:
            conn.close()
    
    def load_daily_data(
        self, 
        symbol: str, 
        start: Optional[str] = None, 
        end: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """加载日线数据"""
        import sqlite3
        
        conn = sqlite3.connect(str(self.db_path))
        try:
            query = "SELECT date, open, high, low, close, volume FROM daily_data WHERE symbol = ?"
            params = [symbol]
            
            if start:
                query += " AND date >= ?"
                params.append(start)
            if end:
                query += " AND date <= ?"
                params.append(end)
            
            query += " ORDER BY date"
            
            df = pd.read_sql_query(query, conn, params=params)
            if df.empty:
                return None
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
        finally:
            conn.close()
    
    def list_symbols(self) -> List[str]:
        """列出所有已存储的股票代码"""
        import sqlite3
        
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.execute("SELECT DISTINCT symbol FROM daily_data ORDER BY symbol")
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def save_backtest_result(self, name: str, result: dict) -> None:
        """保存回测结果"""
        import sqlite3
        
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE,
                    initial_cash REAL,
                    final_value REAL,
                    total_return REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    total_trades INTEGER,
                    daily_values TEXT,
                    trades TEXT,
                    created_at TEXT
                )
            """)
            
            conn.execute("""
                INSERT OR REPLACE INTO backtest_results
                (name, initial_cash, final_value, total_return, max_drawdown, sharpe_ratio, total_trades, daily_values, trades, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                name,
                result.get('initial_cash', 0),
                result.get('final_value', 0),
                result.get('total_return', 0),
                result.get('max_drawdown', 0),
                result.get('sharpe_ratio', 0),
                result.get('total_trades', 0),
                result.get('daily_values').to_json() if 'daily_values' in result else None,
                result.get('trades').to_json() if 'trades' in result else None,
                datetime.now().isoformat()
            ))
            
            conn.commit()
        finally:
            conn.close()
    
    def load_backtest_result(self, name: str) -> Optional[dict]:
        """加载回测结果"""
        import sqlite3
        
        conn = sqlite3.connect(str(self.db_path))
        try:
            df = pd.read_sql_query(
                "SELECT * FROM backtest_results WHERE name = ?",
                conn,
                params=[name]
            )
            
            if df.empty:
                return None
            
            row = df.iloc[0]
            result = {
                'name': row['name'],
                'initial_cash': row['initial_cash'],
                'final_value': row['final_value'],
                'total_return': row['total_return'],
                'max_drawdown': row['max_drawdown'],
                'sharpe_ratio': row['sharpe_ratio'],
                'total_trades': row['total_trades'],
                'daily_values': pd.read_json(row['daily_values']) if pd.notna(row['daily_values']) else None,
                'trades': pd.read_json(row['trades']) if pd.notna(row['trades']) else None,
                'created_at': row['created_at']
            }
            return result
        finally:
            conn.close()


class ParquetStorage:
    """Parquet 数据持久化（适合大量数据）"""
    
    def __init__(self, data_dir: str = "data/parquet"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def save_daily_data(self, symbol: str, df: pd.DataFrame) -> None:
        """保存日线数据到 Parquet"""
        file_path = self.data_dir / f"{symbol.replace('.', '_')}.parquet"
        
        if file_path.exists():
            # 合并现有数据和新数据
            existing_df = pd.read_parquet(file_path)
            combined = pd.concat([existing_df, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=['date'], keep='last')
            combined = combined.sort_values('date').reset_index(drop=True)
            combined.to_parquet(file_path, index=False)
        else:
            df.to_parquet(file_path, index=False)
    
    def load_daily_data(
        self, 
        symbol: str, 
        start: Optional[str] = None, 
        end: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """加载日线数据"""
        file_path = self.data_dir / f"{symbol.replace('.', '_')}.parquet"
        
        if not file_path.exists():
            return None
        
        df = pd.read_parquet(file_path)
        
        if start:
            df = df[df['date'] >= start]
        if end:
            df = df[df['date'] <= end]
        
        return df.reset_index(drop=True) if not df.empty else None
    
    def list_symbols(self) -> List[str]:
        """列出所有已存储的股票代码"""
        symbols = []
        for f in self.data_dir.glob("*.parquet"):
            symbol = f.stem.replace('_', '.')
            symbols.append(symbol)
        return sorted(symbols)


class DataStorage:
    """统一数据存储接口"""
    
    def __init__(self, backend: str = "parquet", **kwargs):
        """
        Args:
            backend: "sqlite" 或 "parquet" (默认parquet，速度更快)
        """
        if backend == "sqlite":
            self.storage = SQLiteStorage(**kwargs)
        elif backend == "parquet":
            self.storage = ParquetStorage(**kwargs)
        else:
            raise ValueError(f"不支持的后端: {backend}")
        
        self.backend = backend
    
    def save_daily_data(self, symbol: str, df: pd.DataFrame) -> None:
        self.storage.save_daily_data(symbol, df)
    
    def load_daily_data(
        self, 
        symbol: str, 
        start: Optional[str] = None, 
        end: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        return self.storage.load_daily_data(symbol, start, end)
    
    # 别名方法，方便使用
    def save_to_parquet(self, df: pd.DataFrame, symbol: str) -> None:
        """保存到Parquet（兼容接口）"""
        if self.backend == "parquet":
            self.storage.save_daily_data(symbol, df)
        else:
            raise ValueError("当前后端不是parquet")
    
    def load_from_parquet(self, symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> Optional[pd.DataFrame]:
        """从Parquet加载（兼容接口）"""
        return self.load_daily_data(symbol, start, end)
    
    def list_symbols(self) -> List[str]:
        return self.storage.list_symbols()
    
    def save_backtest_result(self, name: str, result: dict) -> None:
        if hasattr(self.storage, 'save_backtest_result'):
            self.storage.save_backtest_result(name, result)
        else:
            raise NotImplementedError("Parquet 后端暂不支持保存回测结果")
    
    def load_backtest_result(self, name: str) -> Optional[dict]:
        if hasattr(self.storage, 'load_backtest_result'):
            return self.storage.load_backtest_result(name)
        else:
            raise NotImplementedError("Parquet 后端暂不支持加载回测结果")
