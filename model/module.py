import math

from einops import rearrange, reduce
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

# ============================================================================
# Kronos 基础神经网络模块
# ----------------------------------------------------------------------------
# 本文件提供构建 Kronos 所需的底层组件：
#   - 二值球面量化器 BinarySphericalQuantizer / BSQuantizer
#   - 归一化 RMSNorm、前馈 FeedForward（SwiGLU）
#   - 旋转位置编码 RoPE 与多头（交叉）注意力
#   - 分层嵌入 HierarchicalEmbedding、依赖感知层 DependencyAwareLayer
#   - Transformer 块、双头输出 DualHead、时间嵌入 TemporalEmbedding
# ============================================================================


class DifferentiableEntropyFunction(Function):
    """可微的码本熵计算（自定义 autograd Function），用于 BSQ 的熵惩罚项。

    背景：直接对「码字使用频次」做统计是不可微的（涉及离散索引计数），
    因此这里自定义前向与反向传播，使「码本熵 H」能够回传梯度，从而
    鼓励码本被均匀、充分地使用。
    """
    @staticmethod
    def forward(ctx, zq, basis, K, eps):
        # zq ∈ {-1, +1}，先映射到 {0, 1}
        zb = (zq + 1) / 2
        # 用 basis（2 的幂权重）把每个码字的比特组合折叠成唯一整数索引
        zi = ((zb * basis).sum(-1)).to(torch.int64)
        # 统计每个码字索引出现的次数（散射求和，得到长度为 2^K 的频次向量）
        cnt = torch.scatter_reduce(torch.zeros(2 ** K, device=zq.device, dtype=zq.dtype),
                                   0,
                                   zi.flatten(),
                                   torch.ones_like(zi.flatten()).to(zq.dtype),
                                   'sum')
        # 频次归一化为概率分布（加 eps 防止除零/取对数出错）
        prob = (cnt + eps) / (cnt + eps).sum()
        # 计算香农熵 H = -Σ p·log(p)
        H = -(prob * torch.log(prob)).sum()
        # 保存反向传播所需的中间量
        ctx.save_for_backward(zq, zi, prob)
        ctx.K = K
        return H

    @staticmethod
    def backward(ctx, grad_output):
        zq, zi, prob = ctx.saved_tensors
        # 熵对各码字概率的梯度为 -(log(p) + 1)，再按样本数与位数 K 归一化
        grad_array = -grad_output * (torch.log(prob) + 1) / zi.numel() / ctx.K
        # 按每个位置实际命中的码字索引取回对应梯度，并还原成原始形状
        reord_grad = grad_array[zi.flatten()].reshape(zi.shape)
        # 链式法则：再乘以 zq，得到对输入的梯度
        grad_input = reord_grad.unsqueeze(-1) * zq
        # 只有第一个输入 zq 需要梯度，其余参数返回 None
        return grad_input, None, None, None, None


def codebook_entropy(zq, basis, K, eps=1e-4):
    """码本熵的便捷封装：调用上面可微的熵计算 Function。"""
    return DifferentiableEntropyFunction.apply(zq, basis, K, eps)


class BinarySphericalQuantizer(nn.Module):
    """二值球面量化器（Binary Spherical Quantizer, BSQ）。

    核心思想：把连续向量按每一维的符号量化为 ±1，再做球面（L2）缩放，
    使所有码字均匀分布在单位超球面上。配合「直通估计（STE）」让梯度可回传，
    并通过 commit 损失与熵惩罚共同优化。论文：https://arxiv.org/pdf/2406.07548.pdf
    """
    def __init__(self, embed_dim, beta, gamma0, gamma, zeta,
                 input_format='bchw',
                 soft_entropy=True, group_size=9,
                 persample_entropy_compute='analytical',
                 cb_entropy_compute='group',
                 l2_norm=True,
                 inv_temperature=1):
        """
        Paper link: https://arxiv.org/pdf/2406.07548.pdf
        Here we use the official implementation of the BinarySphericalQuantizer.
        """
        super().__init__()
        self.embed_dim = embed_dim     # 量化向量维度（= 总比特数）
        self.beta = beta  # commit 损失权重（约束编码器输出靠近量化结果）
        self.gamma0 = gamma0  # 逐样本熵的权重
        self.gamma = gamma  # 码本熵的权重
        self.zeta = zeta  # 整体熵惩罚的权重
        self.input_format = input_format
        assert self.embed_dim % group_size == 0, "embed_dim must be divisible by group_size"
        self.num_groups = self.embed_dim // group_size  # 分组数量（用于熵的近似计算）
        self.group_size = group_size                    # 每组比特数
        assert persample_entropy_compute in ['group', 'analytical'], "persample_entropy_compute must be either 'group' or 'analytical'"
        assert cb_entropy_compute in ['group', 'nce'], "cb_entropy_compute must be either 'group' or 'nce'"
        self.persample_entropy_compute = persample_entropy_compute
        self.cb_entropy_compute = cb_entropy_compute
        self.l2_norm = l2_norm                  # 是否做球面（L2）缩放
        self.inv_temperature = inv_temperature  # 逆温度（控制软概率的锐度）

        # basis：把 ±1 比特向量折叠为整数索引时使用的 2 的幂权重（高位在前）
        self.register_buffer('basis', 2 ** torch.arange(embed_dim - 1, -1, -1))
        # group_basis：分组内的折叠权重
        self.register_buffer('group_basis', 2 ** torch.arange(group_size - 1, -1, -1))

        self.num_dimensions = 2 ** embed_dim  # 完整码本大小
        self.bits_per_index = embed_dim

        # 仅保留「分组子码本」即可近似计算熵损失，避免构造 2^embed_dim 的超大码本
        group_codes = torch.arange(2 ** self.group_size)
        group_codebook = self.indexes_to_codes(group_codes).float()[:, -group_size:]
        self.register_buffer('group_codebook', group_codebook, persistent=False)

        self.soft_entropy = soft_entropy  # soft_entropy: Sec 3.2 of https://arxiv.org/pdf/1911.05894.pdf

    def quantize(self, z):
        """对输入向量按符号量化为 ±1，并使用直通估计（STE）保持梯度可回传。"""
        assert z.shape[-1] == self.embed_dim, f"Expected {self.embed_dim} dimensions, got {z.shape[-1]}"

        # 按符号取 +1 / -1
        zhat = torch.where(z > 0,
                           torch.tensor(1, dtype=z.dtype, device=z.device),
                           torch.tensor(-1, dtype=z.dtype, device=z.device))
        # 直通估计：前向用 zhat，反向梯度等同于直接传给 z（(zhat - z).detach() 不参与求导）
        return z + (zhat - z).detach()

    def forward(self, z, collect_metrics=True):
        """前向量化：返回量化向量、损失（commit + 熵惩罚）以及统计指标。"""
        # if self.input_format == 'bchw':
        #     z = rearrange(z, 'b c h w -> b h w c')
        zq = self.quantize(z)  # 量化为 ±1

        # 球面缩放因子：1/sqrt(d)，使码字落在单位球面上
        q_scale = 1. / (self.embed_dim ** 0.5) if self.l2_norm else 1.

        zq = zq * q_scale

        # 推理快速路径：不收集指标时，直接返回量化结果（损失置零）
        if not collect_metrics:
            return zq, zq.new_zeros(()), {}

        # 计算码字索引与分组索引（用于统计/熵计算，detach 避免影响梯度）
        indices = self.codes_to_indexes(zq.detach())
        group_indices = self.codes_to_group_indexes(zq.detach())
        if not self.training:
            used_codes = torch.unique(indices, return_counts=False)  # 评估时统计实际使用的码字
        else:
            used_codes = None

        # 熵惩罚：鼓励逐样本分布有确定性，同时鼓励整体码本被均匀使用
        if self.soft_entropy:
            persample_entropy, cb_entropy, avg_prob = self.soft_entropy_loss(z)
            entropy_penalty = self.gamma0 * persample_entropy - self.gamma * cb_entropy
        else:
            zb_by_sample = ((zq + 1) / 2).reshape(z.shape[0], -1, z.shape[-1]).to(torch.float32)
            persample_entropy = self.get_hard_per_sample_entropy(zb_by_sample)
            cb_entropy = codebook_entropy(zq, self.basis, self.embed_dim)
            entropy_penalty = self.gamma0 * persample_entropy - self.gamma * cb_entropy

        # commit loss
        commit_loss = self.beta * torch.mean(((zq.detach() - z) ** 2).sum(dim=-1))

        # if self.input_format == 'bchw':
        #     zq = rearrange(zq, 'b h w c -> b c h w')

        return (
            zq,
            commit_loss + self.zeta * entropy_penalty / self.inv_temperature,
            {"H": cb_entropy, "used_codes": used_codes, "indices": indices, "group_indices": group_indices,
             "avg_prob": avg_prob}
        )

    def soft_entropy_loss(self, z):
        # if we divide the code in subgroups of size group_size, the codebook will be of size 2 ** group_size
        # the sub-code is the last group_size bits of the full code
        group_code_book = self.group_codebook / (self.embed_dim ** 0.5 if self.l2_norm else 1)
        divided_z = rearrange(z, '... (g c) -> ... g c', c=self.group_size)

        # we calculate the distance between the divided_z and the codebook for each subgroup
        distance = - 2 * torch.einsum('... g c, d c ->... g d', divided_z, group_code_book)
        prob = (-distance * self.inv_temperature).softmax(dim=-1)
        if self.persample_entropy_compute == 'analytical':
            if self.l2_norm:
                p = torch.sigmoid(-4 * z / (self.embed_dim ** 0.5) * self.inv_temperature)
            else:
                p = torch.sigmoid(-4 * z * self.inv_temperature)
            prob = torch.stack([p, 1 - p], dim=-1)
            per_sample_entropy = self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean()
        else:
            per_sample_entropy = self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean()

        # macro average of the probability of each subgroup
        avg_prob = reduce(prob, '... g d ->g d', 'mean')
        codebook_entropy = self.get_entropy(avg_prob, dim=-1, normalize=False)

        # the approximation of the entropy is the sum of the entropy of each subgroup
        return per_sample_entropy, codebook_entropy.sum(), avg_prob

    def get_hard_per_sample_entropy(self, zb_by_sample):
        probs_per_dim = zb_by_sample.sum(1) / zb_by_sample.shape[1]
        persample_entropy = - probs_per_dim * torch.log(probs_per_dim + 1e-8) - (1 - probs_per_dim) * torch.log(1 - probs_per_dim + 1e-8)
        persample_entropy = persample_entropy.sum(-1)
        return persample_entropy.mean()

    def codes_to_indexes(self, zhat):
        """将一个「码字」（±1 比特向量）转换为码本中的整数索引。
        Converts a `code` to an index in the codebook.
        Args:
            zhat: A tensor of shape (B, ..., C) containing the codes. must be in {-1, 1}
        """
        assert zhat.shape[-1] == self.embed_dim, f"Expected {self.embed_dim} dimensions, got {zhat.shape[-1]}"
        # 先把 ±1 映射到 {0,1}，再与 2 的幂权重 basis 内积，得到唯一整数索引
        return ((zhat + 1) / 2 * self.basis).sum(axis=-1).to(torch.int64)

    def codes_to_group_indexes(self, zhat):
        """将一个「码字」按分组转换为多个分组索引。
        Converts a `code` to a list of indexes (in groups) in the codebook.
        Args:
            zhat: A tensor of shape (B, ..., C) containing the codes. must be in {-1, 1}
        """
        # 按 group_size 拆分比特，再在组内折叠为索引
        zhat_in_group = rearrange(zhat, 'b ... (g c) -> b ... g c', c=self.group_size)
        return ((zhat_in_group + 1) / 2 * self.group_basis).sum(axis=-1).to(torch.int64)

    def indexes_to_codes(self, indices):
        """整数索引 -> ±1 码字（codes_to_indexes 的逆运算）。"""
        indices = indices.unsqueeze(-1)
        # 通过 “除以 basis 后取模 2” 逐位还原出 0/1，再映射回 ±1
        codes_non_centered = torch.remainder(
            torch.floor_divide(indices, self.basis), 2
        )
        return codes_non_centered * 2 - 1

    def group_indexes_to_codes(self, group_indices):
        """Inverse of `group_indexes_to_codes`."""
        group_indices = group_indices.unsqueeze(-1)
        codes_non_centered = torch.remainder(
            torch.floor_divide(group_indices, self.group_basis), 2
        )
        codes_non_centered = rearrange(codes_non_centered, 'b ... g c -> b ... (g c)')
        return codes_non_centered * 2 - 1

    def get_entropy(self, count, dim=-1, eps=1e-4, normalize=True):
        if normalize:
            probs = (count + eps) / (count + eps).sum(dim=dim, keepdim=True)
        else:
            probs = count
        H = -(probs * torch.log(probs + 1e-8)).sum(dim=dim)
        return H

    def get_group_codebook_entry(self, group_indices):
        z_q = self.group_indexes_to_codes(group_indices)
        q_scale = 1. / (self.embed_dim ** 0.5) if self.l2_norm else 1.
        z_q = z_q * q_scale
        if self.input_format == 'bchw':
            h, w = int(z_q.shape[1] ** 0.5)
            assert h * w == z_q.shape[1], 'Invalid sequence length'
            z_q = rearrange(z_q, 'b (h w) c -> b c h w', h=h)
        return z_q

    def get_codebook_entry(self, indices):
        z_q = self.indexes_to_codes(indices)
        q_scale = 1. / (self.embed_dim ** 0.5) if self.l2_norm else 1.
        z_q = z_q * q_scale
        if self.input_format == 'bchw':
            h, w = int(z_q.shape[1] ** 0.5)
            assert h * w == z_q.shape[1], 'Invalid sequence length'
            z_q = rearrange(z_q, 'b (h w) c -> b c h w', h=h)
        return z_q


class BSQuantizer(nn.Module):
    """BSQ 量化器包装层：在 BinarySphericalQuantizer 基础上支持分层 token。

    把总码本拆分为 s1（前 s1_bits 位，粗粒度）与 s2（后 s2_bits 位，细粒度）两部分。
    """

    def __init__(self, s1_bits, s2_bits, beta, gamma0, gamma, zeta, group_size):
        super().__init__()
        self.codebook_dim = s1_bits + s2_bits  # 总比特数 = s1 + s2
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.bsq = BinarySphericalQuantizer(self.codebook_dim, beta, gamma0, gamma, zeta, group_size=group_size)

    def bits_to_indices(self, bits):
        """把比特向量（以 0 为阈值判正负）折叠为整数索引（低位在前）。"""
        bits = (bits >= 0).to(torch.long)
        indices = 2 ** torch.arange(
            0,
            bits.shape[-1],
            1,
            dtype=torch.long,
            device=bits.device,
        )
        return (bits * indices).sum(-1)

    def forward(self, z, half=False, collect_metrics=True):
        """先对输入做 L2 归一化，再调用 BSQ 量化。

        half=True 时返回 [s1_indices, s2_indices]（分层 token）；half=False 时返回单一合并索引。
        """
        z = F.normalize(z, dim=-1)  # 投影到单位球面，与 BSQ 的球面假设一致
        quantized, bsq_loss, metrics = self.bsq(z, collect_metrics=collect_metrics)
        if half:
            q_pre = quantized[:, :, :self.s1_bits]    # 前 s1_bits 位 -> s1
            q_post = quantized[:, :, self.s1_bits:]   # 后 s2_bits 位 -> s2
            z_indices = [self.bits_to_indices(q_pre), self.bits_to_indices(q_post)]
        else:
            z_indices = self.bits_to_indices(quantized)
        return bsq_loss, quantized, z_indices


class RMSNorm(torch.nn.Module):
    """均方根归一化（RMSNorm）：相比 LayerNorm 不减均值，计算更轻量。"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # x / sqrt(mean(x^2) + eps)，仅按均方根缩放，不减均值
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 先转 float 计算以保证数值稳定，再转回原类型，最后乘以可学习缩放参数 weight
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    """SwiGLU 风格前馈网络：w2(silu(w1(x)) * w3(x))。"""
    def __init__(self, d_model, ff_dim, ffn_dropout_p=0.0):
        super().__init__()

        self.w1 = nn.Linear(d_model, ff_dim, bias=False)  # 门控分支（过 SiLU 激活）
        self.w3 = nn.Linear(d_model, ff_dim, bias=False)  # 线性分支
        self.w2 = nn.Linear(ff_dim, d_model, bias=False)  # 输出投影回 d_model
        self.ffn_dropout = nn.Dropout(ffn_dropout_p)

    def forward(self, x):
        # SwiGLU：用 SiLU(w1(x)) 作为门控与 w3(x) 逐元素相乘，再投影回原维度
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class RotaryPositionalEmbedding(nn.Module):
    """旋转位置编码（RoPE）：通过旋转查询/键向量注入相对位置信息，带 cos/sin 缓存。"""
    def __init__(self, dim):
        super().__init__()
        # 频率表：不同维度对应不同旋转角速，周期随维度指数增长
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        # 缓存机制：相同序列长度复用已计算的 cos/sin，避免重复计算
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def _update_cos_sin_cache(self, x, seq_len):
        # 仅在序列长度变化时重新计算 cos/sin 缓存
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # 位置 × 频率
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self.cos_cached, self.sin_cached

    def forward(self, q, k):
        # 对 q、k 应用旋转：x·cos + rotate_half(x)·sin
        cos, sin = self._update_cos_sin_cache(q, q.shape[-2])
        return (
            (q * cos) + (self._rotate_half(q) * sin),
            (k * cos) + (self._rotate_half(k) * sin),
        )

    def _rotate_half(self, x):
        # 把后半部分取负后与前半部分交换，实现二维旋转的虚部贡献
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class MultiHeadAttentionWithRoPE(nn.Module):
    """带 RoPE 的多头因果自注意力（用于 Transformer 主干）。"""
    def __init__(self, d_model, n_heads, attn_dropout_p=0.0, resid_dropout_p=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary = RotaryPositionalEmbedding(self.head_dim)
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout_p)

    def forward(self, x, key_padding_mask=None):
        batch_size, seq_len, _ = x.shape

        # 线性投影得到 Q/K/V，并拆分为多头：[B, n_heads, seq_len, head_dim]
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = self.rotary(q, k)  # 注入旋转位置编码

        # 构造 padding 掩码（屏蔽填充位置）
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            attn_mask = attn_mask.expand(-1, self.n_heads, seq_len, -1)  # [batch, n_heads, q_len, k_len]
        else:
            attn_mask = None

        # 缩放点积注意力；is_causal=True 保证自回归的因果性（不能看未来）
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=True
        )

        # 多头合并回 [B, seq_len, d_model]，再输出投影与残差 dropout
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.resid_dropout(self.out_proj(attn_output))


class MultiHeadCrossAttentionWithRoPE(nn.Module):
    """带 RoPE 的多头交叉注意力（用于依赖感知层，query 与 key/value 不同源）。"""
    def __init__(self, d_model, n_heads, attn_dropout_p=0.0, resid_dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary = RotaryPositionalEmbedding(self.head_dim)
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        batch_size, q_len, _ = query.shape
        _, seq_len, _ = key.shape

        q = self.q_proj(query).view(batch_size, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = self.rotary(q, k)

        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(-1, self.n_heads, q_len, -1)
        else:
            attn_mask = None

        is_causal_flag = self.training

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=is_causal_flag
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        return self.resid_dropout(self.out_proj(attn_output))


class HierarchicalEmbedding(nn.Module):
    """分层 token 嵌入：将 s1（粗粒度）与 s2（细粒度）两路嵌入拼接后融合投影。"""
    def __init__(self, s1_bits, s2_bits, d_model=256):
        super().__init__()
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits

        vocab_s1 = 2 ** s1_bits
        vocab_s2 = 2 ** s2_bits

        self.emb_s1 = nn.Embedding(vocab_s1, d_model)
        self.emb_s2 = nn.Embedding(vocab_s2, d_model)
        self.d_model = d_model
        self.fusion_proj = nn.Linear(d_model * 2, d_model)

        nn.init.normal_(self.emb_s1.weight, mean=0, std=d_model ** -0.5)
        nn.init.normal_(self.emb_s2.weight, mean=0, std=d_model ** -0.5)

    def split_token(self, token_ids: torch.Tensor, s2_bits: int):
        """将合成 token 拆分为 (s1_ids, s2_ids)。
        Inputs:
            token_ids (torch.Tensor): Composite token IDs of shape [batch_size, seq_len] or [N], each in range [0, 2^(s1_bits + s2_bits) - 1].
            s2_bits (int): Number of low bits used for the fine token (s2).
        """
        assert isinstance(s2_bits, int) and s2_bits > 0, "s2_bits must be a positive integer"

        t = token_ids.long()
        mask = (1 << s2_bits) - 1
        s2_ids = t & mask           # 取低 s2_bits 位（细粒度）
        s1_ids = t >> s2_bits       # 右移取高位（粗粒度）
        return s1_ids, s2_ids

    def forward(self, token_ids):
        """Inputs:
        token_ids:
            - tuple or list: (s1_ids, s2_ids), each of shape [batch_size, seq_len], or
            - torch.Tensor: composite token IDs of shape [batch_size, seq_len], which will be split into (s1_ids, s2_ids) internally.
        Output: [batch_size, seq_len, d_model]
        """
        # 输入可以是已拆分的 (s1, s2)，也可以是合成 token（需内部拆分）
        if isinstance(token_ids, tuple) or isinstance(token_ids, list):
            s1_ids, s2_ids = token_ids
        else:
            s1_ids, s2_ids = self.split_token(token_ids, self.s2_bits)
        # 分别查表嵌入，乘 sqrt(d_model) 是 Transformer 常用的嵌入缩放
        s1_emb = self.emb_s1(s1_ids) * math.sqrt(self.d_model)
        s2_emb = self.emb_s2(s2_ids) * math.sqrt(self.d_model)
        # 拼接两路嵌入后通过线性层融合为统一表示
        return self.fusion_proj(torch.cat([s1_emb, s2_emb], dim=-1))


class DependencyAwareLayer(nn.Module):
    """依赖感知层：通过交叉注意力让一个子 token（如 s2）感知另一个子 token（如 s1）。"""
    def __init__(self, d_model, n_heads=4, attn_dropout_p=0.0, resid_dropout=0.0):
        super().__init__()
        self.cross_attn = MultiHeadCrossAttentionWithRoPE(d_model, n_heads, attn_dropout_p, resid_dropout)
        self.norm = RMSNorm(d_model)

    def forward(self, hidden_states, sibling_embed, key_padding_mask=None):
        """hidden_states: [batch, seq_len, d_model]
        sibling_embed: Embedding from another subtoken
        """
        # 以另一个子 token 的嵌入作为 query，对当前上下文做交叉注意力
        attn_out = self.cross_attn(
            query=sibling_embed,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask
        )
        # 残差连接 + 归一化
        return self.norm(hidden_states + attn_out)


class TransformerBlock(nn.Module):
    """Transformer 基础块：Pre-Norm 残差结构（自注意力 + 前馈）。"""
    def __init__(self, d_model, n_heads, ff_dim=1024, ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = MultiHeadAttentionWithRoPE(d_model, n_heads, attn_dropout_p, resid_dropout_p)
        self.norm2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, ff_dim, ffn_dropout_p)

    def forward(self, x, key_padding_mask=None):
        # 第一个子层：Pre-Norm + 自注意力 + 残差
        residual = x
        x = self.norm1(x)
        attn_out = self.self_attn(x, key_padding_mask=key_padding_mask)
        x = residual + attn_out

        # 第二个子层：Pre-Norm + 前馈 + 残差
        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = residual + ffn_out
        return x


class DualHead(nn.Module):
    """双头输出：分别预测 s1 与 s2 的 logits，并提供 s1/s2 交叉熵损失计算。"""
    def __init__(self, s1_bits, s2_bits, d_model):
        super().__init__()
        self.vocab_s1 = 2 ** s1_bits
        self.vocab_s2 = 2 ** s2_bits
        self.proj_s1 = nn.Linear(d_model, self.vocab_s1)
        self.proj_s2 = nn.Linear(d_model, self.vocab_s2)

    def compute_loss(self, s1_logits, s2_logits, s1_targets, s2_targets, padding_mask=None):
        """计算 s1 与 s2 的交叉熵损失，取平均作为总损失。"""
        if padding_mask is not None:
            # 仅在非填充位置（padding_mask == 0）上计算损失
            valid_mask = (padding_mask == 0)
            s1_logits = s1_logits[valid_mask]
            s2_logits = s2_logits[valid_mask]
            s1_targets = s1_targets[valid_mask]
            s2_targets = s2_targets[valid_mask]
            ce_s1 = F.cross_entropy(s1_logits, s1_targets)
            ce_s2 = F.cross_entropy(s2_logits, s2_targets)
        else:
            ce_s1 = F.cross_entropy(s1_logits.reshape(-1, self.vocab_s1), s1_targets.reshape(-1))
            ce_s2 = F.cross_entropy(s2_logits.reshape(-1, self.vocab_s2), s2_targets.reshape(-1))
        ce_loss = (ce_s1 + ce_s2) / 2
        return ce_loss, ce_s1, ce_s2

    def forward(self, x):
        # 预测 s1（粗粒度）logits
        return self.proj_s1(x)

    def cond_forward(self, x2):
        # 在 s1 条件下预测 s2（细粒度）logits
        return self.proj_s2(x2)


class FixedEmbedding(nn.Module):
    """固定（不可学习）的正弦位置嵌入，用于时间特征。"""
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        # 预计算正余/余弦位置编码表，并冻结为不参与训练的参数
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)  # 偶数维用 sin
        w[:, 1::2] = torch.cos(position * div_term)  # 奇数维用 cos

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        # detach 确保梯度不回传到固定嵌入
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """时间特征嵌入：对分钟/小时/星期/日/月分别嵌入后求和，可选可学习或固定。"""
    def __init__(self, d_model, learn_pe):
        super(TemporalEmbedding, self).__init__()

        # 各时间字段的取值范围（用作各自嵌入表的词表大小）
        minute_size = 60
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        # learn_pe=True 时使用可学习嵌入，否则使用固定正弦嵌入
        Embed = FixedEmbedding if not learn_pe else nn.Embedding
        self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        # 输入 x: [batch, seq_len, 5]，最后一维依次为 [minute, hour, weekday, day, month]
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 0])
        hour_x = self.hour_embed(x[:, :, 1])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 3])
        month_x = self.month_embed(x[:, :, 4])

        # 将五类时间嵌入求和作为最终时间特征
        return hour_x + weekday_x + day_x + month_x + minute_x








