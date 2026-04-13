# 工具函数模块

import pandas as pd
from typing import List, Dict


def format_number(num: float, decimals: int = 2) -> str:
    """格式化数字"""
    return f"{num:,.{decimals}f}"


def format_percent(num: float, decimals: int = 2) -> str:
    """格式化百分比"""
    return f"{num * 100:.{decimals}f}%"


def calculate_drawdown(values: List[float]) -> List[float]:
    """计算回撤序列"""
    drawdowns = []
    peak = values[0]
    
    for value in values:
        if value > peak:
            peak = value
        drawdown = (value - peak) / peak
        drawdowns.append(drawdown)
    
    return drawdowns


def annual_return(total_return: float, days: int) -> float:
    """计算年化收益率"""
    if days <= 0:
        return 0.0
    return (1 + total_return) ** (365 / days) - 1


def volatility(returns: List[float]) -> float:
    """计算波动率"""
    import numpy as np
    return np.std(returns) * (252 ** 0.5)  # 年化