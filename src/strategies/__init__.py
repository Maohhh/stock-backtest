# 策略模块
from .base import BaseStrategy, BuyAndHoldStrategy, MovingAverageCrossStrategy
from .rsi_strategy import RSIStrategy
from .macd_strategy import MACDStrategy
from .bollinger_strategy import BollingerStrategy
from .kdj_strategy import KDJStrategy
from .atr_strategy import ATRStrategy
from .main_force_strategy import MainForceResonanceStrategy
from .futures_xs_reversal import FuturesXSReversalStrategy
from .trend_following import TrendFollowingStrategy
from .spread_reversion import PairReversionStrategy

__all__ = [
    'BaseStrategy',
    'BuyAndHoldStrategy',
    'MovingAverageCrossStrategy',
    'RSIStrategy',
    'MACDStrategy',
    'BollingerStrategy',
    'KDJStrategy',
    'ATRStrategy',
    'MainForceResonanceStrategy',
    'FuturesXSReversalStrategy',
    'TrendFollowingStrategy',
    'PairReversionStrategy',
]