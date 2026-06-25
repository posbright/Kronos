"""因子元数据：为每个 tech_*/fin_* 因子提供中文名称、释义、分类与方向先验。

中文说明：
    本模块是「因子调参工作台」的知识库，把原本只有英文列名的因子补全为：
      - name_cn  中文名称
      - desc_cn  一句话释义（含计算/含义/常见用法）
      - category 大类（用于分组、聚合、同类冗余分析）
      - bias     方向先验：+1 偏多（数值越高越偏多头）/ -1 偏空 / 0 视情况（震荡/均值回复）
                 注意：这是「教科书先验」，真实多空方向以 factor_analysis 的经验 IC 为准。

    分类体系（category）：
      技术面：trend 趋势 / momentum 动量 / volatility 波动 / volume 量能
      基本面：valuation 估值 / profit 盈利 / growth 成长 / solvency 偿债 /
              operation 营运 / cashflow 现金流 / expense 费用 / scale 规模

用法：
    from finetune_csv.app.factor_meta import get_meta, group_by_category, all_meta
    m = get_meta("tech_macd")             # -> dict 或 None（未知列回退）
    groups = group_by_category(cols)      # -> {category: [col, ...]}

    python -m finetune_csv.app.factor_meta --smoke
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

# 大类的中文名与所属面（technical / fundamental）
CATEGORY_INFO: Dict[str, Dict[str, str]] = {
    "trend":      {"name_cn": "趋势类",   "side": "technical"},
    "momentum":   {"name_cn": "动量类",   "side": "technical"},
    "volatility": {"name_cn": "波动类",   "side": "technical"},
    "volume":     {"name_cn": "量能类",   "side": "technical"},
    "valuation":  {"name_cn": "估值类",   "side": "fundamental"},
    "profit":     {"name_cn": "盈利类",   "side": "fundamental"},
    "growth":     {"name_cn": "成长类",   "side": "fundamental"},
    "solvency":   {"name_cn": "偿债类",   "side": "fundamental"},
    "operation":  {"name_cn": "营运类",   "side": "fundamental"},
    "cashflow":   {"name_cn": "现金流类", "side": "fundamental"},
    "expense":    {"name_cn": "费用类",   "side": "fundamental"},
    "scale":      {"name_cn": "规模类",   "side": "fundamental"},
}

# 每个因子的元数据。bias: +1 偏多 / -1 偏空 / 0 视情况。
_META: Dict[str, Dict[str, object]] = {
    # ---------- 技术面 · 趋势 ----------
    "tech_macd":   {"name_cn": "MACD 离差值(DIF)", "category": "trend", "bias": 1,
                    "desc_cn": "快慢均线之差(EMA12-EMA26)，DIF 上穿 0 轴/信号线视为多头趋势确立。"},
    "tech_macds":  {"name_cn": "MACD 信号线(DEA)", "category": "trend", "bias": 1,
                    "desc_cn": "DIF 的 9 日 EMA，与 DIF 的金叉/死叉是经典买卖信号。"},
    "tech_macdh":  {"name_cn": "MACD 柱(MACD)", "category": "trend", "bias": 1,
                    "desc_cn": "(DIF-DEA)×2，柱状由负转正/放大表示多头动能增强。"},
    "tech_dma":    {"name_cn": "DMA 平行线差", "category": "trend", "bias": 1,
                    "desc_cn": "短期与长期均线差值，反映中短期趋势的强弱与拐点。"},
    "tech_sar":    {"name_cn": "SAR 抛物线转向", "category": "trend", "bias": 1,
                    "desc_cn": "停损转向指标，价格在 SAR 之上为多头、之下为空头。"},
    "tech_trix":   {"name_cn": "TRIX 三重平滑均线", "category": "trend", "bias": 1,
                    "desc_cn": "三重指数平滑变动率，过滤短期噪声，适合中长期趋势确认。"},
    "tech_pdi":    {"name_cn": "+DI 上升方向线", "category": "trend", "bias": 1,
                    "desc_cn": "DMI 中的上升动向指标，+DI 在 -DI 之上偏多。"},
    "tech_mdi":    {"name_cn": "-DI 下降方向线", "category": "trend", "bias": -1,
                    "desc_cn": "DMI 中的下降动向指标，-DI 走高表示下行动能增强。"},
    "tech_adx":    {"name_cn": "ADX 平均趋向指数", "category": "trend", "bias": 0,
                    "desc_cn": "衡量趋势强度(不分方向)，>25 趋势明显、<20 震荡盘整。"},
    "tech_adxr":   {"name_cn": "ADXR 趋向评估线", "category": "trend", "bias": 0,
                    "desc_cn": "ADX 的平滑评估值，用于判断趋势是否持续。"},

    # ---------- 技术面 · 动量 / 超买超卖 ----------
    "tech_kdjk":   {"name_cn": "KDJ-K 值", "category": "momentum", "bias": 0,
                    "desc_cn": "随机指标快线，>80 超买、<20 超卖，金叉偏多。"},
    "tech_kdjd":   {"name_cn": "KDJ-D 值", "category": "momentum", "bias": 0,
                    "desc_cn": "随机指标慢线，K 上穿 D 为买入信号。"},
    "tech_kdjj":   {"name_cn": "KDJ-J 值", "category": "momentum", "bias": 0,
                    "desc_cn": "3K-2D，灵敏度最高，常超出 0~100 区间提示极端。"},
    "tech_rsi_6":  {"name_cn": "RSI 相对强弱(6日)", "category": "momentum", "bias": 0,
                    "desc_cn": "短周期 RSI，>70 超买、<30 超卖，敏感但噪声大。"},
    "tech_rsi_12": {"name_cn": "RSI 相对强弱(12日)", "category": "momentum", "bias": 0,
                    "desc_cn": "中短周期 RSI，平衡灵敏度与稳定性。"},
    "tech_rsi":    {"name_cn": "RSI 相对强弱(默认)", "category": "momentum", "bias": 0,
                    "desc_cn": "默认周期 RSI，衡量上涨力量占比。"},
    "tech_rsi_24": {"name_cn": "RSI 相对强弱(24日)", "category": "momentum", "bias": 0,
                    "desc_cn": "长周期 RSI，更平滑，适合中线趋势。"},
    "tech_cci":    {"name_cn": "CCI 顺势指标", "category": "momentum", "bias": 0,
                    "desc_cn": "价格偏离统计均值的程度，>+100 强势、<-100 弱势。"},
    "tech_roc":    {"name_cn": "ROC 变动率", "category": "momentum", "bias": 1,
                    "desc_cn": "当前价相对 N 日前的涨跌幅，>0 上行动量。"},
    "tech_rocma":  {"name_cn": "ROCMA 变动率均线", "category": "momentum", "bias": 1,
                    "desc_cn": "ROC 的移动平均，平滑动量信号。"},
    "tech_wr_6":   {"name_cn": "威廉指标 WR(6日)", "category": "momentum", "bias": 0,
                    "desc_cn": "短周期 W%R，反向刻度，越接近 0 越超买。"},
    "tech_wr_10":  {"name_cn": "威廉指标 WR(10日)", "category": "momentum", "bias": 0,
                    "desc_cn": "中周期 W%R，衡量收盘价在区间高低位置。"},
    "tech_wr_14":  {"name_cn": "威廉指标 WR(14日)", "category": "momentum", "bias": 0,
                    "desc_cn": "标准周期 W%R，超买超卖判别。"},
    "tech_psy":    {"name_cn": "PSY 心理线", "category": "momentum", "bias": 0,
                    "desc_cn": "N 日内上涨天数占比，反映市场情绪，>75 偏热。"},
    "tech_psyma":  {"name_cn": "PSYMA 心理线均线", "category": "momentum", "bias": 0,
                    "desc_cn": "PSY 的移动平均，平滑情绪波动。"},

    # ---------- 技术面 · 波动 ----------
    "tech_boll":   {"name_cn": "BOLL 布林中轨", "category": "volatility", "bias": 1,
                    "desc_cn": "中轨为 N 日均线，价格在中轨上方偏强。"},
    "tech_boll_ub":{"name_cn": "BOLL 布林上轨", "category": "volatility", "bias": 0,
                    "desc_cn": "中轨+k×标准差，触及上轨提示超买/压力。"},
    "tech_boll_lb":{"name_cn": "BOLL 布林下轨", "category": "volatility", "bias": 0,
                    "desc_cn": "中轨-k×标准差，触及下轨提示超卖/支撑。"},
    "tech_atr":    {"name_cn": "ATR 平均真实波幅", "category": "volatility", "bias": 0,
                    "desc_cn": "衡量波动幅度(不分方向)，常用于仓位与止损设置。"},

    # ---------- 技术面 · 量能 ----------
    "tech_obv":    {"name_cn": "OBV 能量潮", "category": "volume", "bias": 1,
                    "desc_cn": "按涨跌累计成交量，量价齐升验证趋势。"},
    "tech_vr":     {"name_cn": "VR 成交量比率", "category": "volume", "bias": 0,
                    "desc_cn": "上涨与下跌成交量之比，衡量多空资金强弱。"},
    "tech_mfi":    {"name_cn": "MFI 资金流量指标", "category": "volume", "bias": 0,
                    "desc_cn": "结合价格与成交量的 RSI，>80 资金超买、<20 超卖。"},

    # ---------- 基本面 · 盈利 ----------
    "fin_eps":              {"name_cn": "每股收益 EPS", "category": "profit", "bias": 1,
                             "desc_cn": "净利润/总股本，每股盈利能力，越高越好。"},
    "fin_net_profit":       {"name_cn": "净利润", "category": "profit", "bias": 1,
                             "desc_cn": "归属股东的利润总额，盈利规模。"},
    "fin_roe":              {"name_cn": "净资产收益率 ROE", "category": "profit", "bias": 1,
                             "desc_cn": "净利润/净资产，股东回报核心指标。"},
    "fin_roa":              {"name_cn": "总资产收益率 ROA", "category": "profit", "bias": 1,
                             "desc_cn": "净利润/总资产，资产整体盈利效率。"},
    "fin_gross_margin":     {"name_cn": "毛利率", "category": "profit", "bias": 1,
                             "desc_cn": "(营收-成本)/营收，反映产品议价与成本控制。"},
    "fin_net_profit_margin":{"name_cn": "净利率", "category": "profit", "bias": 1,
                             "desc_cn": "净利润/营收，最终盈利转化能力。"},

    # ---------- 基本面 · 成长 ----------
    "fin_revenue_yoy":      {"name_cn": "营收同比增速", "category": "growth", "bias": 1,
                             "desc_cn": "营业收入同比增长率，成长性核心。"},
    "fin_net_profit_yoy":   {"name_cn": "净利同比增速", "category": "growth", "bias": 1,
                             "desc_cn": "净利润同比增长率，盈利成长性。"},
    "fin_rd_ratio":         {"name_cn": "研发费用占比", "category": "growth", "bias": 1,
                             "desc_cn": "研发支出/营收，反映创新投入强度。"},
    "fin_rd_expense":       {"name_cn": "研发费用", "category": "growth", "bias": 1,
                             "desc_cn": "研发支出绝对额，规模化创新投入。"},

    # ---------- 基本面 · 估值/价值 ----------
    "fin_bps":              {"name_cn": "每股净资产 BPS", "category": "valuation", "bias": 1,
                             "desc_cn": "净资产/总股本，每股账面价值，安全边际参考。"},

    # ---------- 基本面 · 现金流 ----------
    "fin_ocfps":            {"name_cn": "每股经营现金流", "category": "cashflow", "bias": 1,
                             "desc_cn": "经营现金净流量/总股本，盈利质量与造血能力。"},

    # ---------- 基本面 · 规模 ----------
    "fin_revenue":          {"name_cn": "营业收入", "category": "scale", "bias": 1,
                             "desc_cn": "主营业务收入规模。"},

    # ---------- 基本面 · 偿债/杠杆 ----------
    "fin_asset_liability_ratio":{"name_cn": "资产负债率", "category": "solvency", "bias": -1,
                             "desc_cn": "总负债/总资产，杠杆水平，过高偿债风险大。"},
    "fin_current_ratio":    {"name_cn": "流动比率", "category": "solvency", "bias": 1,
                             "desc_cn": "流动资产/流动负债，短期偿债能力。"},
    "fin_quick_ratio":      {"name_cn": "速动比率", "category": "solvency", "bias": 1,
                             "desc_cn": "(流动资产-存货)/流动负债，更严格的短期偿债。"},

    # ---------- 基本面 · 营运 ----------
    "fin_total_asset_turnover":{"name_cn": "总资产周转率", "category": "operation", "bias": 1,
                             "desc_cn": "营收/总资产，资产运营效率。"},
    "fin_inventory_turnover":{"name_cn": "存货周转率", "category": "operation", "bias": 1,
                             "desc_cn": "营业成本/平均存货，存货管理效率。"},
    "fin_receivable_turnover":{"name_cn": "应收账款周转率", "category": "operation", "bias": 1,
                             "desc_cn": "营收/平均应收，回款效率与信用管理。"},

    # ---------- 基本面 · 费用 ----------
    "fin_admin_expense":    {"name_cn": "管理费用", "category": "expense", "bias": -1,
                             "desc_cn": "管理活动支出，占比过高侵蚀利润。"},
    "fin_selling_expense":  {"name_cn": "销售费用", "category": "expense", "bias": -1,
                             "desc_cn": "营销/渠道支出，需结合营收增长评估效率。"},
    "fin_financial_expense":{"name_cn": "财务费用", "category": "expense", "bias": -1,
                             "desc_cn": "利息等融资成本，过高反映债务压力。"},
}


def get_meta(col: str) -> Optional[Dict[str, object]]:
    """返回某列的元数据；未知列回退为一个基于前缀的占位元数据。"""
    if col in _META:
        m = dict(_META[col])
        m["col"] = col
        m["category_cn"] = CATEGORY_INFO.get(str(m["category"]), {}).get("name_cn", "其他")
        m["side"] = CATEGORY_INFO.get(str(m["category"]), {}).get("side", "other")
        return m
    if col.startswith("tech_"):
        return {"col": col, "name_cn": col, "category": "momentum", "category_cn": "动量类",
                "side": "technical", "bias": 0, "desc_cn": "技术指标（未登记释义）。"}
    if col.startswith("fin_"):
        return {"col": col, "name_cn": col, "category": "profit", "category_cn": "盈利类",
                "side": "fundamental", "bias": 0, "desc_cn": "基本面指标（未登记释义）。"}
    return None


def all_meta(cols: List[str]) -> List[Dict[str, object]]:
    """对一组列批量返回元数据（保持输入顺序）。"""
    return [get_meta(c) for c in cols if get_meta(c) is not None]


def group_by_category(cols: List[str]) -> Dict[str, List[str]]:
    """把列按 category 分组，返回 {category: [col, ...]}（保持类内输入顺序）。"""
    groups: Dict[str, List[str]] = {}
    for c in cols:
        m = get_meta(c)
        if m is None:
            continue
        groups.setdefault(str(m["category"]), []).append(c)
    return groups


def _smoke() -> None:
    cols = list(_META.keys())
    # 1) 每个因子都有完整字段
    for c in cols:
        m = get_meta(c)
        assert m and m["name_cn"] and m["desc_cn"], f"{c} 元数据缺失"
        assert m["category"] in CATEGORY_INFO, f"{c} 分类非法: {m['category']}"
        assert m["bias"] in (-1, 0, 1), f"{c} bias 非法: {m['bias']}"
    # 2) 分组覆盖全部
    groups = group_by_category(cols)
    total = sum(len(v) for v in groups.values())
    assert total == len(cols), f"分组数量不一致 {total} != {len(cols)}"
    # 3) 未知列回退
    assert get_meta("tech_unknown_x")["category"] == "momentum"
    assert get_meta("fin_unknown_x")["side"] == "fundamental"
    assert get_meta("foobar") is None
    print(f"[smoke] factor_meta 通过：{len(cols)} 因子，{len(groups)} 个大类",
          {k: len(v) for k, v in groups.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="因子元数据库")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
    else:
        ap.print_help()
