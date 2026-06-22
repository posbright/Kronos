# 04. 核心 API 参考

本篇汇总 `model` 包对外暴露的核心类与方法。源码：[model/kronos.py](../model/kronos.py)、[model/module.py](../model/module.py)。

## 4.1 导出入口

[model/__init__.py](../model/__init__.py) 对外导出：

```python
from model import Kronos, KronosTokenizer, KronosPredictor
```

另提供 `get_model_class(model_name)`，按名称返回模型类（`kronos_tokenizer` / `kronos` / `kronos_predictor`）。

## 4.2 KronosTokenizer

继承自 `nn.Module` 与 `PyTorchModelHubMixin`，支持 `from_pretrained` / `save_pretrained`。

### 构造参数

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `d_in` | int | 输入特征维度（如 OHLCV+amount=6）。 |
| `d_model` | int | 模型隐藏维度。 |
| `n_heads` | int | 注意力头数。 |
| `ff_dim` | int | 前馈网络维度。 |
| `n_enc_layers` | int | 编码器层数。 |
| `n_dec_layers` | int | 解码器层数。 |
| `ffn_dropout_p` | float | 前馈 dropout 概率。 |
| `attn_dropout_p` | float | 注意力 dropout 概率。 |
| `resid_dropout_p` | float | 残差 dropout 概率。 |
| `s1_bits` | int | 粗粒度（pre）token 位数。 |
| `s2_bits` | int | 细粒度（post）token 位数。 |
| `beta` | float | BSQ commit loss 权重。 |
| `gamma0`, `gamma`, `zeta` | float | BSQ 熵惩罚相关权重。 |
| `group_size` | int | BSQ 分组大小（用于熵近似计算）。 |

### 主要方法

| 方法 | 签名 | 返回 |
| --- | --- | --- |
| `forward` | `forward(x)` | `((z_pre, z), bsq_loss, quantized, z_indices)` |
| `encode` | `encode(x, half=False)` | token 索引（`half=True` 返回 `[s1, s2]`） |
| `decode` | `decode(x, half=False)` | 重建的特征张量 `[B, T, d_in]` |
| `indices_to_bits` | `indices_to_bits(x, half=False)` | ±1 缩放后的比特向量 |

## 4.3 Kronos

继承自 `nn.Module` 与 `PyTorchModelHubMixin`。

### 构造参数

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `s1_bits`, `s2_bits` | int | 粗/细粒度 token 位数（须与 Tokenizer 一致）。 |
| `n_layers` | int | Transformer 层数。 |
| `d_model` | int | 嵌入与隐藏维度。 |
| `n_heads` | int | 注意力头数。 |
| `ff_dim` | int | 前馈网络维度。 |
| `ffn_dropout_p` / `attn_dropout_p` / `resid_dropout_p` | float | 各类 dropout 概率。 |
| `token_dropout_p` | float | token 嵌入 dropout。 |
| `learn_te` | bool | 是否使用可学习时间嵌入。 |

### 主要方法

| 方法 | 说明 |
| --- | --- |
| `forward(s1_ids, s2_ids, stamp=None, padding_mask=None, use_teacher_forcing=False, s1_targets=None)` | 训练前向，返回 `(s1_logits, s2_logits)`。 |
| `decode_s1(s1_ids, s2_ids, stamp=None, padding_mask=None)` | 仅预测 s1，返回 `(s1_logits, context)`。 |
| `decode_s2(context, s1_ids, padding_mask=None)` | 基于上下文与 s1 预测 s2，返回 `s2_logits`。 |

## 4.4 KronosPredictor

高层封装，负责数据准备、归一化、调用自回归推理、反归一化。

### 构造函数

```python
KronosPredictor(model, tokenizer, device=None, max_context=512, clip=5)
```

| 参数 | 说明 |
| --- | --- |
| `model` | `Kronos` 实例。 |
| `tokenizer` | `KronosTokenizer` 实例。 |
| `device` | 计算设备；为 `None` 时自动选择 `cuda` / `mps` / `cpu`。 |
| `max_context` | 最大上下文长度（须 ≤ 模型上限）。 |
| `clip` | 归一化后裁剪阈值，抑制异常值。 |

内部约定的列名：价格列 `['open','high','low','close']`、成交量列 `volume`、成交额列 `amount`、时间特征 `['minute','hour','weekday','day','month']`。

### predict 方法

```python
predict(df, x_timestamp, y_timestamp, pred_len,
        T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=True) -> pd.DataFrame
```

| 参数 | 说明 |
| --- | --- |
| `df` | 历史 K 线 DataFrame，须含 OHLC 四列；缺 `volume`/`amount` 自动补 0。 |
| `x_timestamp` | 历史时间戳 Series。 |
| `y_timestamp` | 未来预测时间戳 Series。 |
| `pred_len` | 预测步数。 |
| `T` / `top_k` / `top_p` | 采样控制参数。 |
| `sample_count` | 预测路径数（取均值）。 |
| `verbose` | 是否显示进度条。 |

**处理流程**：校验列 → 补全 volume/amount → 检查 NaN → 提取时间特征 → 按列做 z-score 归一化并裁剪 → 调用 `generate` → 反归一化 → 组装为带 `y_timestamp` 索引的 DataFrame。

返回：包含 `open/high/low/close/volume/amount` 预测值的 DataFrame。

### predict_batch 方法

```python
predict_batch(df_list, x_timestamp_list, y_timestamp_list, pred_len,
              T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=True) -> List[pd.DataFrame]
```

并行预测多组序列。**要求所有序列历史长度一致、预测长度一致**，否则抛出 `ValueError`。返回与输入顺序一致的预测 DataFrame 列表。

### generate 方法（内部）

```python
generate(x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, verbose)
```

把 numpy 数组转为张量并调用 `auto_regressive_inference`，返回最后 `pred_len` 步的预测（numpy）。

## 4.5 模块级函数

| 函数 | 说明 |
| --- | --- |
| `auto_regressive_inference(...)` | 自回归生成核心循环（见 [03_模型架构.md](./03_模型架构.md) 3.4 节）。 |
| `top_k_top_p_filtering(logits, top_k, top_p, ...)` | Top-k / 核采样过滤。 |
| `sample_from_logits(logits, temperature, top_k, top_p, sample_logits)` | 按温度采样。 |
| `calc_time_stamps(x_timestamp)` | 从时间戳提取时间特征 DataFrame。 |

下一篇：[05_微调指南.md](./05_微调指南.md)
