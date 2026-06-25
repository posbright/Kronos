"""把预训练 KronosTokenizer 的输入维度从 d_in 扩展到 d_in + k（方案 A）。

中文说明：
    Kronos 的 ``KronosTokenizer`` 输入侧只有 ``embed`` / ``head`` 两个线性层与 ``d_in`` 绑定。
    本脚本读取预训练目录里的 ``config.json``（含全部 BSQ 超参），仅覆盖 ``d_in`` 后重建一个
    新 tokenizer，并把可复用的中间层权重原样移植、把 ``embed`` / ``head`` 的前 ``old_d_in``
    维从预训练权重继承（新增维度保持随机初始化），最后保存为新的预训练目录。

用法：
    # 真实使用（把 PE/PB 等 2 个因子拼到 6 维之后 -> d_in=8）
    python expand_tokenizer.py \
        --pretrained /path/to/Kronos-Tokenizer-base \
        --save-dir   /path/to/Kronos-Tokenizer-base-d8 \
        --k-extra    2

    # 冒烟自测（无需任何预训练权重，自动造一个临时 tokenizer 跑通全流程）
    python expand_tokenizer.py --smoke
"""

import argparse
import inspect
import json
import os
import sys
import tempfile

# 确保可以 `from model import KronosTokenizer`（本文件位于 finetune_csv/ 下）
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
from model import KronosTokenizer  # noqa: E402


def expand_tokenizer(pretrained: str, save_dir: str, k_extra: int) -> str:
    """读取预训练 tokenizer，扩展 d_in 并移植权重，保存到 save_dir。返回 save_dir。"""
    if k_extra <= 0:
        raise ValueError(f"k_extra 必须为正整数，收到 {k_extra}")

    # 1) 加载预训练 tokenizer（d_in=6）
    tok_old = KronosTokenizer.from_pretrained(pretrained)
    old_d_in = tok_old.d_in
    new_d_in = old_d_in + k_extra

    # 2) 从 config.json 读取「完整且真实」的超参，仅覆盖 d_in。
    #    ★ 关键：BSQ 超参（beta/gamma0/gamma/zeta/group_size 等）会影响量化器内部结构
    #    （如 group_size 需满足 embed_dim % group_size == 0），绝不能手填猜测值。
    config_path = os.path.join(pretrained, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    valid_keys = set(inspect.signature(KronosTokenizer.__init__).parameters) - {"self"}
    cfg = {k: v for k, v in cfg.items() if k in valid_keys}
    cfg["d_in"] = new_d_in
    tok_new = KronosTokenizer(**cfg)

    # 3) 拷贝可复用的中间层（除 embed/head 外全部同名同形）
    old_sd, new_sd = tok_old.state_dict(), tok_new.state_dict()
    copied = 0
    for name, w in old_sd.items():
        if name in new_sd and new_sd[name].shape == w.shape:
            new_sd[name] = w.clone()
            copied += 1

    # 4) 部分继承 embed / head 的前 old_d_in 维，新增维度保持随机初始化
    #    若不同 Kronos 版本改了输入层命名，提前报清晰错误而非 KeyError。
    required = ["embed.weight", "embed.bias", "head.weight", "head.bias"]
    missing = [k for k in required if k not in old_sd or k not in new_sd]
    if missing:
        raise KeyError(
            f"tokenizer 缺少预期的输入层权重键 {missing}；当前 KronosTokenizer 结构可能已变更，"
            f"请检查 embed/head 的命名后再移植。")
    #    embed.weight: [d_model, d_in] -> 复制前 old_d_in 列；embed.bias: [d_model] -> 直接复制
    new_sd["embed.weight"][:, :old_d_in] = old_sd["embed.weight"]
    new_sd["embed.bias"] = old_sd["embed.bias"].clone()
    #    head.weight: [d_in, d_model] -> 复制前 old_d_in 行；head.bias: [d_in] -> 复制前 old_d_in 个
    new_sd["head.weight"][:old_d_in, :] = old_sd["head.weight"]
    new_sd["head.bias"][:old_d_in] = old_sd["head.bias"]

    tok_new.load_state_dict(new_sd)
    os.makedirs(save_dir, exist_ok=True)
    tok_new.save_pretrained(save_dir)

    print(f"[expand_tokenizer] d_in {old_d_in} -> {new_d_in}; "
          f"复用中间层张量 {copied}/{len(old_sd)} 个；已保存到 {save_dir}")
    return save_dir


def _make_dummy_tokenizer(save_dir: str) -> None:
    """造一个微型 tokenizer 并保存，仅用于冒烟自测。"""
    torch.manual_seed(0)
    tok = KronosTokenizer(
        d_in=6, d_model=32, n_heads=4, ff_dim=64, n_enc_layers=2, n_dec_layers=2,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        s1_bits=4, s2_bits=4, beta=1.0, gamma0=1.0, gamma=1.0, zeta=1.0, group_size=4,
    )
    tok.save_pretrained(save_dir)


def _smoke_test() -> None:
    """无需真实权重的端到端冒烟测试。"""
    with tempfile.TemporaryDirectory() as tmp:
        pretrained = os.path.join(tmp, "tok_base")
        save_dir = os.path.join(tmp, "tok_base_d8")
        _make_dummy_tokenizer(pretrained)

        expand_tokenizer(pretrained, save_dir, k_extra=2)

        # 重新加载扩展后的 tokenizer 并做一次 d_in=8 的前向
        tok = KronosTokenizer.from_pretrained(save_dir)
        assert tok.d_in == 8, f"期望 d_in=8，实际 {tok.d_in}"
        x = torch.randn(1, 10, tok.d_in)
        with torch.no_grad():
            idx = tok.encode(x, half=True)
        assert isinstance(idx, (list, tuple)) and len(idx) == 2, "encode 应返回 (s1, s2)"
        print("[smoke] expand_tokenizer 通过：d_in=8 重建 + 权重移植 + encode 前向均正常")


def main() -> None:
    parser = argparse.ArgumentParser(description="扩展 KronosTokenizer 的输入维度 d_in（方案 A）")
    parser.add_argument("--pretrained", help="预训练 tokenizer 目录（含 config.json）")
    parser.add_argument("--save-dir", help="扩展后 tokenizer 的保存目录")
    parser.add_argument("--k-extra", type=int, default=2, help="新增因子个数 k（d_in -> d_in+k）")
    parser.add_argument("--smoke", action="store_true", help="运行无需权重的冒烟自测")
    args = parser.parse_args()

    if args.smoke:
        _smoke_test()
        return

    if not args.pretrained or not args.save_dir:
        parser.error("非 --smoke 模式下必须提供 --pretrained 与 --save-dir")
    expand_tokenizer(args.pretrained, args.save_dir, args.k_extra)


if __name__ == "__main__":
    main()
