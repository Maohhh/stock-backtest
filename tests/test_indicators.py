"""
技术指标模块单元测试
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from indicators import sma, ema, macd, rsi, bollinger_bands, kdj, atr


class TestIndicators(unittest.TestCase):
    
    def setUp(self):
        """创建测试数据"""
        np.random.seed(42)
        n = 100
        
        # 生成模拟股票数据
        base_price = 100
        prices = [base_price]
        for i in range(1, n):
            change = np.random.normal(0, 2)
            prices.append(max(prices[-1] + change, 1))  # 确保价格为正
        
        self.df = pd.DataFrame({
            'open': prices,
            'high': [p + abs(np.random.normal(1, 0.5)) for p in prices],
            'low': [p - abs(np.random.normal(1, 0.5)) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, n)
        })
    
    def test_sma_basic(self):
        """测试SMA基本功能"""
        result = sma(self.df, period=20)
        
        # 检查结果类型
        self.assertIsInstance(result, pd.Series)
        # 检查长度与输入相同
        self.assertEqual(len(result), len(self.df))
        # 检查没有NaN (因为有min_periods=1)
        self.assertFalse(result.isna().any())
    
    def test_sma_calculation(self):
        """测试SMA计算正确性"""
        # 使用简单数据测试
        simple_df = pd.DataFrame({'close': [1, 2, 3, 4, 5]})
        result = sma(simple_df, period=3)
        
        # 手动计算验证
        expected = pd.Series([1.0, 1.5, 2.0, 3.0, 4.0], name='close')
        pd.testing.assert_series_equal(result, expected)
    
    def test_ema_basic(self):
        """测试EMA基本功能"""
        result = ema(self.df, period=20)
        
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.df))
        self.assertFalse(result.isna().any())
    
    def test_macd_basic(self):
        """测试MACD基本功能"""
        result = macd(self.df)
        
        # 检查结果类型
        self.assertIsInstance(result, pd.DataFrame)
        # 检查返回的列
        self.assertIn('DIF', result.columns)
        self.assertIn('DEA', result.columns)
        self.assertIn('MACD', result.columns)
        # 检查长度
        self.assertEqual(len(result), len(self.df))
    
    def test_macd_calculation(self):
        """测试MACD计算关系"""
        result = macd(self.df)
        
        # MACD柱状图 = (DIF - DEA) * 2
        expected_macd = (result['DIF'] - result['DEA']) * 2
        expected_macd.name = 'MACD'
        pd.testing.assert_series_equal(result['MACD'], expected_macd)
    
    def test_rsi_basic(self):
        """测试RSI基本功能"""
        result = rsi(self.df, period=14)
        
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.df))
        # RSI应在0-100范围内 (允许前几个值为NaN)
        valid_rsi = result.dropna()
        self.assertTrue((valid_rsi >= 0).all() and (valid_rsi <= 100).all())
    
    def test_rsi_range(self):
        """测试RSI范围限制"""
        # 使用极端价格变化测试
        extreme_df = pd.DataFrame({
            'close': [100] + [100 + i*10 for i in range(1, 20)]  # 持续上涨
        })
        result = rsi(extreme_df, period=14)
        
        # RSI应接近100但不超出 (允许前几个值为NaN)
        valid_rsi = result.dropna()
        self.assertTrue((valid_rsi >= 0).all() and (valid_rsi <= 100).all())
    
    def test_bollinger_bands_basic(self):
        """测试布林带基本功能"""
        result = bollinger_bands(self.df)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('middle', result.columns)
        self.assertIn('upper', result.columns)
        self.assertIn('lower', result.columns)
        self.assertEqual(len(result), len(self.df))
    
    def test_bollinger_relationship(self):
        """测试布林带各线关系"""
        result = bollinger_bands(self.df)
        
        # 上轨 >= 中轨 >= 下轨 (允许NaN值)
        valid_data = result.dropna()
        self.assertTrue((valid_data['upper'] >= valid_data['middle']).all())
        self.assertTrue((valid_data['middle'] >= valid_data['lower']).all())
    
    def test_kdj_basic(self):
        """测试KDJ基本功能"""
        result = kdj(self.df)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('K', result.columns)
        self.assertIn('D', result.columns)
        self.assertIn('J', result.columns)
        self.assertEqual(len(result), len(self.df))
    
    def test_kdj_relationship(self):
        """测试KDJ计算关系"""
        result = kdj(self.df)
        
        # J = 3K - 2D
        expected_j = 3 * result['K'] - 2 * result['D']
        expected_j.name = 'J'
        pd.testing.assert_series_equal(result['J'], expected_j)
    
    def test_atr_basic(self):
        """测试ATR基本功能"""
        result = atr(self.df)
        
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.df))
        # ATR应为正值
        self.assertTrue((result > 0).all())
    
    def test_atr_positive(self):
        """测试ATR始终为正"""
        result = atr(self.df)
        
        # ATR代表波幅，应该始终为正
        self.assertTrue((result >= 0).all())
    
    def test_column_validation(self):
        """测试列名验证"""
        bad_df = pd.DataFrame({'price': [1, 2, 3]})
        
        with self.assertRaises(ValueError):
            sma(bad_df, column='close')  # close列不存在
        
        with self.assertRaises(ValueError):
            kdj(bad_df)  # 缺少high/low/close
        
        with self.assertRaises(ValueError):
            atr(bad_df)  # 缺少high/low/close
    
    def test_parameter_customization(self):
        """测试参数自定义"""
        # 测试不同周期参数
        sma_10 = sma(self.df, period=10)
        sma_30 = sma(self.df, period=30)
        
        # 周期短的SMA应该对价格变化更敏感 (使用均值绝对偏差)
        sma_10_mad = (sma_10 - sma_10.mean()).abs().mean()
        sma_30_mad = (sma_30 - sma_30.mean()).abs().mean()
        self.assertGreater(sma_10_mad, sma_30_mad * 0.5)  # 允许一定容差


if __name__ == '__main__':
    unittest.main()
