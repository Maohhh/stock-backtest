"""
LON 龙系长线策略 (Long-line Energy Strategy)

以 LON 长线指标为基础的拐点策略：

- 买入：LON 在 0 轴下方（空头能量区），且 LON 柱体长度连续两天变短
        —— 空头能量衰竭，预示下跌动能减弱、可能见底。
- 卖出：LON 在 0 轴上方（多头能量区），且 LON 柱体长度连续两天变短
        —— 多头能量衰竭，预示上涨动能减弱、可能见顶。

"柱体长度连续两天变短" 定义为：
    len[t] < len[t-1] 且 len[t-1] < len[t-2]
其中 len = abs(LON)。
"""

from typing import Dict
import pandas as pd
from .base import BaseStrategy
from ..indicators.lon import lon


class LONStrategy(BaseStrategy):
    """
    LON 龙系长线策略

    策略逻辑：
    - LON < 0 且柱体长度连续两天变短 -> 买入（空头能量衰竭，潜在底部）
    - LON > 0 且柱体长度连续两天变短 -> 卖出（多头能量衰竭，潜在顶部）

    参数:
        diff_period: LON DIFF 周期，默认 10
        dea_period:  LON DEA 周期，默认 20
        ma_period:   LONMA 周期，默认 6
        amount:      每次交易股数，默认 100
    """

    def __init__(self,
                 diff_period: int = 10,
                 dea_period: int = 20,
                 ma_period: int = 6,
                 amount: int = 100):
        super().__init__("LONStrategy")
        self.diff_period = diff_period
        self.dea_period = dea_period
        self.ma_period = ma_period
        self.amount = amount

    def set_params(self, **kwargs):
        super().set_params(**kwargs)
        for key in ('diff_period', 'dea_period', 'ma_period', 'amount'):
            if key in kwargs:
                setattr(self, key, kwargs[key])

    def on_bar(self, context: Dict) -> Dict:
        data = context['data']

        # 至少需要 3 根 LON 柱才能判断"连续两天变短"，加上指标预热
        if len(data) < max(self.dea_period, 3) + 3:
            return None

        lon_df = lon(
            data,
            diff_period=self.diff_period,
            dea_period=self.dea_period,
            ma_period=self.ma_period,
        )

        lon_now = lon_df['lon'].iloc[-1]
        len0 = lon_df['lon_len'].iloc[-1]   # 当前
        len1 = lon_df['lon_len'].iloc[-2]   # 前一天
        len2 = lon_df['lon_len'].iloc[-3]   # 前两天

        # 柱体长度连续两天变短
        shrinking_two_days = (len0 < len1) and (len1 < len2)
        if not shrinking_two_days:
            return None

        # 0 轴下方：空头能量衰竭 -> 买入
        if lon_now < 0:
            return {'direction': 'buy', 'amount': self.amount}

        # 0 轴上方：多头能量衰竭 -> 卖出
        if lon_now > 0:
            return {'direction': 'sell', 'amount': self.amount}

        return None
