"""在原始 Kronos 主模型上新增一路「加性因子条件嵌入」（方案 B）。

中文说明：
    Kronos 主模型本就有一条不经过 tokenizer 的条件通路 ``TemporalEmbedding``（把时间特征投影到
    d_model 后与 token 嵌入相加）。本模块用子类 ``KronosWithFactor`` 再加一路 ``factor_emb``
    （``nn.Linear(k, d_model)``），把因子向量同样投影到 d_model 与 token 嵌入相加。

    关键设计：``factor_emb`` 权重 / 偏置零初始化 —— 训练起点等价于原始 Kronos，训练中再逐步学到
    因子的边际贡献，收敛更稳、不易破坏预训练能力。Tokenizer 完全不动（d_in 仍为 6）。

用法：
    from finetune_csv.factor_model import load_with_factor
    model = load_with_factor("/path/to/Kronos-base", factor_dim=4)
    # forward 时多传一个 factor=[B, T, k]
    s1_logits, s2_logits = model(s1_ids, s2_ids, stamp=stamp, factor=factor)

    # 冒烟自测（无需任何预训练权重）
    python factor_model.py --smoke
"""

import argparse
import os
import sys

# 确保可以 `from model.kronos import Kronos`（本文件位于 finetune_csv/ 下）
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from model.kronos import Kronos  # noqa: E402


class KronosWithFactor(Kronos):
    """在原始 Kronos 上新增一路加性因子条件嵌入。"""

    def init_factor(self, factor_dim: int) -> None:
        """挂上一路零初始化的因子嵌入 ``factor_emb = nn.Linear(factor_dim, d_model)``。

        Args:
            factor_dim: 因子向量维度 k（须为正整数，且与数据集/推理时一致）。
        """
        if not isinstance(factor_dim, int) or factor_dim <= 0:
            raise ValueError(f"factor_dim 必须为正整数，收到 {factor_dim!r}")
        # 单独初始化，避免影响 from_pretrained 的权重加载（先 from_pretrained 再调用本方法）
        self.factor_dim = factor_dim
        self.factor_emb = nn.Linear(factor_dim, self.d_model)
        # 关键：权重 / 偏置置 0 -> 初始等价于原始 Kronos，训练中再学增量
        nn.init.zeros_(self.factor_emb.weight)
        nn.init.zeros_(self.factor_emb.bias)

    def _check_factor(self, factor) -> None:
        """校验传入因子张量的形状与已初始化的 factor_emb 是否匹配。"""
        if not hasattr(self, "factor_emb"):
            raise RuntimeError("使用 factor 前请先调用 init_factor(factor_dim)")
        if factor.dim() != 3:
            raise ValueError(f"factor 期望 3 维 [B, T, k]，收到 {tuple(factor.shape)}")
        if factor.size(-1) != self.factor_dim:
            raise ValueError(
                f"factor 末维 {factor.size(-1)} 与 init_factor 的 factor_dim={self.factor_dim} 不一致")

    def forward(self, s1_ids, s2_ids, stamp=None, factor=None,
                padding_mask=None, use_teacher_forcing=False, s1_targets=None):
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            x = x + self.time_emb(stamp)
        if factor is not None:                       # 新增：因子条件旁路
            self._check_factor(factor)
            x = x + self.factor_emb(factor)
        x = self.token_drop(x)

        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)
        x = self.norm(x)

        s1_logits = self.head(x)
        if use_teacher_forcing:
            sibling_embed = self.embedding.emb_s1(s1_targets)
        else:
            s1_probs = F.softmax(s1_logits.detach(), dim=-1)
            sample_s1_ids = torch.multinomial(
                s1_probs.view(-1, self.s1_vocab_size), 1).view(s1_ids.shape)
            sibling_embed = self.embedding.emb_s1(sample_s1_ids)

        x2 = self.dep_layer(x, sibling_embed, key_padding_mask=padding_mask)
        s2_logits = self.head.cond_forward(x2)
        return s1_logits, s2_logits

    def decode_s1(self, s1_ids, s2_ids, stamp=None, factor=None, padding_mask=None):
        """与基类 decode_s1 一致，但额外注入因子条件旁路（供 FactorPredictor 自回归推理用）。

        factor 为 None 时行为与基类完全一致（零初始化时也等价于原始 Kronos）。
        """
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            x = x + self.time_emb(stamp)
        if factor is not None:
            self._check_factor(factor)
            x = x + self.factor_emb(factor)
        x = self.token_drop(x)

        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)
        x = self.norm(x)

        s1_logits = self.head(x)
        return s1_logits, x


def load_with_factor(pretrained_predictor: str, factor_dim: int) -> "KronosWithFactor":
    """加载预训练主模型权重，并挂上零初始化的因子嵌入。"""
    model = KronosWithFactor.from_pretrained(pretrained_predictor)
    model.init_factor(factor_dim)
    return model


def load_factor_model(model_dir: str, factor_dim: int) -> "KronosWithFactor":
    """重新加载已训练好的 KronosWithFactor（含 factor_emb 学到的权重）。

    关键坑：from_pretrained 依据 __init__ 配置重建模型，此时 factor_emb 尚不存在，
    因此 factor_emb.weight/bias 不会被加载；若随后直接 init_factor 又会把它清零，
    导致「学到的因子权重丢失」。本函数在 init_factor 之后，再从检查点把 factor_emb
    的权重补回（strict=False 仅匹配存在的键）。

    Args:
        model_dir:  best_model 目录（含 model.safetensors 或 pytorch_model.bin）。
        factor_dim: 因子维度（须与训练时一致）。
    """
    model = KronosWithFactor.from_pretrained(model_dir)
    model.init_factor(factor_dim)

    st_path = os.path.join(model_dir, "model.safetensors")
    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.exists(st_path):
        from safetensors.torch import load_file
        state = load_file(st_path)
    elif os.path.exists(bin_path):
        state = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"{model_dir} 缺少 model.safetensors / pytorch_model.bin")

    factor_state = {k: v for k, v in state.items() if k.startswith("factor_emb.")}
    if not factor_state:
        raise KeyError(f"{model_dir} 检查点缺少 factor_emb.*，可能不是因子模型。")
    model.load_state_dict(factor_state, strict=False)
    return model


def _make_tiny_model(factor_dim: int) -> "KronosWithFactor":
    """构造一个微型 KronosWithFactor，仅用于冒烟自测。"""
    torch.manual_seed(0)
    model = KronosWithFactor(
        s1_bits=4, s2_bits=4, n_layers=2, d_model=32, n_heads=4, ff_dim=64,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        token_dropout_p=0.0, learn_te=True,
    )
    model.init_factor(factor_dim)
    return model.eval()


def _smoke_test() -> None:
    """无需真实权重的冒烟测试：验证零初始化等价性与带因子前向形状。"""
    k = 4
    model = _make_tiny_model(factor_dim=k)

    B, T = 1, 12
    s1 = torch.randint(0, 16, (B, T))
    s2 = torch.randint(0, 16, (B, T))
    stamp = torch.zeros(B, T, 5).long()
    factor = torch.randn(B, T, k)

    with torch.no_grad():
        # 固定随机种子保证内部 s1 采样一致，便于比较
        torch.manual_seed(1)
        out_none = model(s1, s2, stamp=stamp, factor=None)[0]
        torch.manual_seed(1)
        out_fac = model(s1, s2, stamp=stamp, factor=factor)[0]

    assert torch.allclose(out_none, out_fac, atol=1e-6), \
        "factor_emb 零初始化时，传入因子不应改变输出（等价性应成立）"
    assert out_fac.shape == (B, T, model.s1_vocab_size), f"输出形状异常: {out_fac.shape}"
    print("[smoke] factor_model 通过：零初始化等价性成立，带因子前向形状正确",
          tuple(out_fac.shape))


def main() -> None:
    parser = argparse.ArgumentParser(description="带因子条件旁路的 Kronos 主模型（方案 B）")
    parser.add_argument("--smoke", action="store_true", help="运行无需权重的冒烟自测")
    args = parser.parse_args()
    if args.smoke:
        _smoke_test()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
