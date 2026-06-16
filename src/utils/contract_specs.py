"""期货合约规格与成本模型(单一数据源, 供各回测脚本共用)。

期货成本必须按合约规格算, 不能用统一"价格基点"——低价高乘数品种(如玉米)会被严重高估。
单边成本(价格单位) = 往返手续费/手 ÷ 乘数 + 单边滑点(tick数) × tick。
短线真正的成本主因是滑点(跨买卖价差), 不是手续费; 故 slippage_ticks 是核心旋钮。

tick/mult/comm_rt 为近似值, 实盘以各品种交易所与券商参数为准。
"""

# product -> (中文名, 板块, tick最小变动, mult合约乘数, comm_rt往返手续费元/手)
PRODUCT_SPEC = {
    # 农产品
    "C":  dict(name="玉米",   sector="农产品",     tick=1.0,  mult=10,   comm_rt=2.4),
    "CS": dict(name="淀粉",   sector="农产品",     tick=1.0,  mult=10,   comm_rt=3.0),
    "M":  dict(name="豆粕",   sector="农产品",     tick=1.0,  mult=10,   comm_rt=3.0),
    "Y":  dict(name="豆油",   sector="农产品",     tick=2.0,  mult=10,   comm_rt=5.0),
    "OI": dict(name="菜油",   sector="农产品",     tick=1.0,  mult=10,   comm_rt=4.0),
    "RM": dict(name="菜粕",   sector="农产品",     tick=1.0,  mult=10,   comm_rt=3.0),
    "P":  dict(name="棕榈",   sector="农产品",     tick=2.0,  mult=10,   comm_rt=5.0),
    # 黑色/工业品
    "RB": dict(name="螺纹钢", sector="黑色/工业品", tick=1.0,  mult=10,   comm_rt=4.0),
    "HC": dict(name="热卷",   sector="黑色/工业品", tick=1.0,  mult=10,   comm_rt=4.0),
    "I":  dict(name="铁矿石", sector="黑色/工业品", tick=0.5,  mult=100,  comm_rt=14.0),
    "J":  dict(name="焦炭",   sector="黑色/工业品", tick=0.5,  mult=100,  comm_rt=14.0),
    # 有色
    "CU": dict(name="沪铜",   sector="有色",       tick=10.0, mult=5,    comm_rt=13.0),
    "AL": dict(name="沪铝",   sector="有色",       tick=5.0,  mult=5,    comm_rt=6.0),
    "ZN": dict(name="沪锌",   sector="有色",       tick=5.0,  mult=5,    comm_rt=6.0),
    "NI": dict(name="沪镍",   sector="有色",       tick=10.0, mult=1,    comm_rt=12.0),
    # 贵金属
    "AU": dict(name="沪金",   sector="贵金属",     tick=0.02, mult=1000, comm_rt=20.0),
    "AG": dict(name="沪银",   sector="贵金属",     tick=1.0,  mult=15,   comm_rt=10.0),
    # 能化
    "TA": dict(name="PTA",   sector="能化",       tick=2.0,  mult=5,    comm_rt=6.0),
    "MA": dict(name="甲醇",   sector="能化",       tick=1.0,  mult=10,   comm_rt=4.0),
    "PP": dict(name="聚丙烯", sector="能化",       tick=1.0,  mult=5,    comm_rt=4.0),
    "EG": dict(name="乙二醇", sector="能化",       tick=1.0,  mult=10,   comm_rt=6.0),
    "FU": dict(name="燃油",   sector="能化",       tick=1.0,  mult=10,   comm_rt=2.0),
    # 股指
    "IF": dict(name="沪深300", sector="股指",      tick=0.2,  mult=300,  comm_rt=26.0),
    "IH": dict(name="上证50",  sector="股指",      tick=0.2,  mult=300,  comm_rt=26.0),
    "IC": dict(name="中证500", sector="股指",      tick=0.2,  mult=200,  comm_rt=30.0),
    "IM": dict(name="中证1000", sector="股指",     tick=0.2,  mult=200,  comm_rt=40.0),
}


def cost_per_side(product: str, slippage_ticks: float = 1.0) -> float:
    """单边成本(价格单位) = 往返手续费/2/乘数 + 滑点tick数 × tick。

    slippage_ticks: 单边滑点(以tick计)。市价单约 1, 被动限价成交约 0。
    返回 None 表示无该品种规格。
    """
    spec = PRODUCT_SPEC.get(product.upper())
    if spec is None:
        return None
    return (spec["comm_rt"] / 2.0) / spec["mult"] + slippage_ticks * spec["tick"]


def roundtrip_yuan(product: str, slippage_ticks: float = 1.0) -> float:
    """往返成本(元/手), 仅用于展示。"""
    spec = PRODUCT_SPEC[product.upper()]
    return cost_per_side(product, slippage_ticks) * 2.0 * spec["mult"]
