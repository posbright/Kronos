"""因子权重分配可视化。

KronosWithFactor 的 factor_emb 是一层 nn.Linear(factor_dim -> d_model)，其权重矩阵
形状为 [d_model, factor_dim]。每一列对应「一个因子」注入主干的方式，列的 L2 范数越大，
说明该因子对模型隐状态的影响越强 —— 以此作为「因子重要性 / 权重分配」的直观度量。

提供：
    - load_factor_emb_weight(model_dir): 从检查点直接读取 factor_emb.weight。
    - factor_importance(weight, names): 计算每个因子的 L2 重要性及归一化占比。
    - plot_factor_weights(importance, out_path): 条形图 + JSON 落盘。
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ._fonts import setup_cjk_fonts  # noqa: E402

setup_cjk_fonts()


def load_factor_emb_weight(model_dir: str) -> np.ndarray:
    """从模型检查点目录读取 factor_emb.weight，返回 [d_model, factor_dim] 的 numpy。

    支持 safetensors（model.safetensors）与 PyTorch（pytorch_model.bin）两种格式。
    """
    key = "factor_emb.weight"
    st_path = os.path.join(model_dir, "model.safetensors")
    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.exists(st_path):
        from safetensors.torch import load_file
        state = load_file(st_path)
    elif os.path.exists(bin_path):
        import torch
        state = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"未在 {model_dir} 找到 model.safetensors 或 pytorch_model.bin")

    if key not in state:
        raise KeyError(f"检查点缺少 {key}，该模型可能不是 KronosWithFactor 或未调用 init_factor。")
    w = state[key]
    return w.detach().cpu().numpy() if hasattr(w, "detach") else np.asarray(w)


def factor_importance(weight: np.ndarray,
                      names: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """计算每个因子的 L2 重要性与归一化占比。

    Args:
        weight: [d_model, factor_dim] 的 factor_emb 权重矩阵。
        names:  因子名列表（长度需等于 factor_dim）；缺省用 factor_0..k。

    Returns:
        {因子名: {"l2": 范数, "share": 占比}}，按 l2 降序。
    """
    weight = np.asarray(weight, dtype=np.float64)
    if weight.ndim != 2:
        raise ValueError(f"weight 应为二维 [d_model, factor_dim]，实际 {weight.shape}")
    k = weight.shape[1]
    if names is None:
        names = [f"factor_{i}" for i in range(k)]
    if len(names) != k:
        raise ValueError(f"names 长度 {len(names)} 与 factor_dim {k} 不一致")

    l2 = np.linalg.norm(weight, axis=0)  # 每列范数 -> 每个因子
    total = float(l2.sum()) or 1.0
    items = sorted(zip(names, l2), key=lambda t: t[1], reverse=True)
    return {n: {"l2": float(v), "share": float(v) / total} for n, v in items}


def plot_factor_weights(importance: Dict[str, Dict[str, float]],
                        out_path: str,
                        title: str = "因子权重分配（factor_emb 列范数）",
                        top_n: Optional[int] = None) -> str:
    """画因子重要性条形图，并把数据同名落盘为 JSON。

    Returns:
        out_path（PNG）。
    """
    items = list(importance.items())
    if top_n is not None:
        items = items[:top_n]
    names = [n for n, _ in items]
    shares = [d["share"] for _, d in items]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(names) + 1)))
    y = np.arange(len(names))
    ax.barh(y, shares, color="#1f77b4", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("归一化占比 (share)")
    ax.set_title(title, fontsize=13)
    for yi, s in zip(y, shares):
        ax.text(s, yi, f" {s*100:.1f}%", va="center", fontsize=8)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    json_path = os.path.splitext(out_path)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(importance, f, ensure_ascii=False, indent=2)
    return out_path
