"""matplotlib 中文字体配置。

DejaVu Sans 不含 CJK 字形，直接绘制中文会显示为方框。此处在导入时探测系统中常见的
中文字体（Windows: 微软雅黑 / 黑体；其它平台的常见开源中文字体），设置为 sans-serif
首选，并关闭 Unicode 负号以避免坐标轴负号显示异常。探测失败时静默退回默认字体。
"""

from __future__ import annotations

import matplotlib

# 候选中文字体（按优先级），覆盖 Windows / Linux / macOS 常见安装。
_CJK_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "SimSun", "KaiTi",
    "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei",
    "Arial Unicode MS", "PingFang SC", "Heiti SC",
]

_DONE = False


def setup_cjk_fonts() -> None:
    """把首个可用的中文字体设为 matplotlib sans-serif 首选（只生效一次）。"""
    global _DONE
    if _DONE:
        return
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((name for name in _CJK_CANDIDATES if name in available), None)
    if chosen is not None:
        base = matplotlib.rcParams.get("font.sans-serif", [])
        matplotlib.rcParams["font.sans-serif"] = [chosen] + [b for b in base if b != chosen]
    matplotlib.rcParams["axes.unicode_minus"] = False  # 正常显示负号
    _DONE = True
