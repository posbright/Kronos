<div align="center">
  <h2><b>Kronos：面向金融市场语言的基础模型 </b></h2>
</div>


<div align="center">

</a> 
<a href="https://huggingface.co/NeoQuasar"> 
<img src="https://img.shields.io/badge/🤗-Hugging_Face-yellow" alt="Hugging Face"> 
</a> 
<a href="https://shiyu-coder.github.io/Kronos-demo/"> <img src="https://img.shields.io/badge/🚀-Live_Demo-brightgreen" alt="Live Demo"> </a>
<a href="https://github.com/shiyu-coder/Kronos/graphs/commit-activity"> 
<img src="https://img.shields.io/github/last-commit/shiyu-coder/Kronos?color=blue" alt="Last Commit"> 
</a> 
<a href="https://github.com/shiyu-coder/Kronos/stargazers"> 
<img src="https://img.shields.io/github/stars/shiyu-coder/Kronos?color=lightblue" alt="GitHub Stars"> 
</a> 
<a href="https://github.com/shiyu-coder/Kronos/network/members"> 
<img src="https://img.shields.io/github/forks/shiyu-coder/Kronos?color=yellow" alt="GitHub Forks"> 
</a> 
<a href="./LICENSE"> 
<img src="https://img.shields.io/github/license/shiyu-coder/Kronos?color=green" alt="License"> 
</a>

</div>

<p align="center">

<img src="./figures/logo.png" width="100">

</p>

> Kronos 是**首个面向金融蜡烛图（K 线）的开源基础模型**，
> 在来自全球 **45+ 交易所** 的数据上完成预训练。


## 📰 新闻动态
*   🚩 **[2025.11.10]** Kronos 已被 AAAI 2026 接收。
*   🚩 **[2025.08.17]** 我们已发布微调脚本！欢迎使用它们将 Kronos 适配到你自己的任务。
*   🚩 **[2025.08.02]** 我们的论文已发布在 [arXiv](https://arxiv.org/abs/2508.02739)！

<p align="center">

## 📜 简介

**Kronos** 是一个**解码器（decoder-only）架构的基础模型家族**，专门针对金融市场的「语言」——K 线序列进行预训练。与通用时间序列基础模型（TSFM）不同，Kronos 专为应对金融数据高噪声、独特的特性而设计。它采用一套新颖的两阶段框架：
1. 一个专用的 Tokenizer（分词器/量化器）首先将连续、多维的 K 线数据（OHLCV：开盘价、最高价、最低价、收盘价、成交量）量化为**分层离散 Token（hierarchical discrete tokens）**。
2. 随后在这些 Token 上预训练一个大型自回归 Transformer，使其能够作为服务于多种量化任务的统一模型。

<p align="center">
    <img src="figures/overview.png" alt="" align="center" width="700px" />
</p>

## ✨ 在线 Demo 
我们搭建了一个在线 Demo 用于可视化 Kronos 的预测结果。该网页展示了对 **BTC/USDT** 交易对未来 24 小时的预测。

**👉 [点此访问在线 Demo](https://shiyu-coder.github.io/Kronos-demo/)** 

## 📦 模型库（Model Zoo） 
我们发布了一系列容量各异的预训练模型，以适应不同的计算与应用需求。所有模型均可从 Hugging Face Hub 便捷获取。

| 模型         | Tokenizer                                                                       | 上下文长度 | 参数量  | 是否开源                                                               |
|--------------|---------------------------------------------------------------------------------| -------------- | ------ |---------------------------------------------------------------------------|
| Kronos-mini  | [Kronos-Tokenizer-2k](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-2k)     | 2048           | 4.1M   | ✅ [NeoQuasar/Kronos-mini](https://huggingface.co/NeoQuasar/Kronos-mini)  |
| Kronos-small | [Kronos-Tokenizer-base](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base) | 512            | 24.7M  | ✅ [NeoQuasar/Kronos-small](https://huggingface.co/NeoQuasar/Kronos-small) |
| Kronos-base  | [Kronos-Tokenizer-base](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base) | 512            | 102.3M | ✅ [NeoQuasar/Kronos-base](https://huggingface.co/NeoQuasar/Kronos-base)   |
| Kronos-large | [Kronos-Tokenizer-base](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base) | 512            | 499.2M | ❌                                                                         |


## 🚀 快速开始

### 安装

1. 安装 Python 3.10+，然后安装依赖：

```shell
pip install -r requirements.txt
```

### 📈 进行预测

使用 `KronosPredictor` 类进行预测非常简单。它负责处理数据预处理、归一化、预测以及反归一化，使你只需几行代码即可从原始数据得到预测结果。

**重要提示**：`Kronos-small` 与 `Kronos-base` 的 `max_context`（最大上下文）为 **512**。这是模型一次可处理的最大序列长度。为获得最佳性能，建议你的输入数据长度（即 `lookback`）不要超过该限制。对于更长的上下文，`KronosPredictor` 会自动进行截断处理。

下面是完成你第一次预测的分步指南。

#### 1. 加载 Tokenizer 与模型

首先，从 Hugging Face Hub 加载一个预训练的 Kronos 模型及其对应的 Tokenizer。

```python
from model import Kronos, KronosTokenizer, KronosPredictor

# 从 Hugging Face Hub 加载
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
```

#### 2. 实例化 Predictor

创建一个 `KronosPredictor` 实例，传入模型、Tokenizer 以及目标设备。

```python
# 初始化预测器
predictor = KronosPredictor(model, tokenizer, max_context=512)
```

#### 3. 准备输入数据

`predict` 方法需要三个主要输入：
-   `df`：包含历史 K 线数据的 pandas DataFrame，必须包含 `['open', 'high', 'low', 'close']` 列；`volume` 和 `amount` 为可选项。
-   `x_timestamp`：与 `df` 中历史数据对应的时间戳 pandas Series。
-   `y_timestamp`：你希望预测的未来各时间点的时间戳 pandas Series。

```python
import pandas as pd

# 加载你的数据
df = pd.read_csv("./data/XSHG_5min_600977.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

# 定义上下文窗口与预测长度
lookback = 400
pred_len = 120

# 为预测器准备输入
x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = df.loc[:lookback-1, 'timestamps']
y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']
```

#### 4. 生成预测 

调用 `predict` 方法生成预测。你可以通过 `T`、`top_p`、`sample_count` 等参数控制采样过程，实现概率化预测。

```python
# 生成预测
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,          # 采样温度
    top_p=0.9,      # 核采样（nucleus sampling）概率
    sample_count=1  # 生成并取平均的预测路径数量
)

print("预测数据前几行：")
print(pred_df.head())
```

`predict` 方法返回一个 pandas DataFrame，包含 `open`、`high`、`low`、`close`、`volume` 与 `amount` 的预测值，索引为你提供的 `y_timestamp`。

为高效处理多条时间序列，Kronos 提供了 `predict_batch` 方法，可同时对多个数据集进行并行预测。当你需要一次性预测多个标的或多个时间段时，这一功能尤其有用。

```python
# 为批量预测准备多个数据集
df_list = [df1, df2, df3]  # DataFrame 列表
x_timestamp_list = [x_ts1, x_ts2, x_ts3]  # 历史时间戳列表
y_timestamp_list = [y_ts1, y_ts2, y_ts3]  # 未来时间戳列表

# 生成批量预测
pred_df_list = predictor.predict_batch(
    df_list=df_list,
    x_timestamp_list=x_timestamp_list,
    y_timestamp_list=y_timestamp_list,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True
)

# pred_df_list 中的预测结果顺序与输入一致
for i, pred_df in enumerate(pred_df_list):
    print(f"序列 {i} 的预测：")
    print(pred_df.head())
```

**批量预测的重要要求：**
- 所有序列必须具有相同的历史长度（lookback 窗口）
- 所有序列必须具有相同的预测长度（`pred_len`）
- 每个 DataFrame 必须包含必需列：`['open', 'high', 'low', 'close']`
- `volume` 与 `amount` 列为可选项，缺失时将用 0 填充

`predict_batch` 方法利用 GPU 并行能力实现高效处理，并自动为每条序列独立完成归一化与反归一化。

#### 5. 示例与可视化

如需完整、可运行的脚本（包含数据加载、预测与绘图），请参见 [`examples/prediction_example.py`](examples/prediction_example.py)。

运行该脚本将生成一张对比真实数据与模型预测的图表，如下所示：

<p align="center">
    <img src="figures/prediction_example.png" alt="Forecast Example" align="center" width="600px" />
</p>

此外，我们还提供了一个在没有成交量（Volume）与成交额（Amount）数据情况下进行预测的脚本，详见 [`examples/prediction_wo_vol_example.py`](examples/prediction_wo_vol_example.py)。


## 🔧 在你自己的数据上微调（以 A 股市场为例）

我们提供了一套完整的流程，用于在你自己的数据集上微调 Kronos。作为示例，我们演示了如何使用 [Qlib](https://github.com/microsoft/qlib) 准备中国 A 股市场数据并进行简单回测。

> **免责声明：** 本流程仅作为演示，用于说明微调过程。它是一个简化示例，并非生产级别的量化交易系统。一个稳健的量化策略需要更复杂的技术（如投资组合优化、风险因子中性化）才能获得稳定的超额收益（alpha）。

微调流程分为四个主要步骤：

1.  **配置**：设置路径与超参数。
2.  **数据准备**：使用 Qlib 处理并切分你的数据。
3.  **模型微调**：微调 Tokenizer 与 Predictor 模型。
4.  **回测**：评估微调后模型的表现。

### 前置条件

1.  首先，确保已安装 `requirements.txt` 中的所有依赖。
2.  本流程依赖 `qlib`，请安装：
    ```shell
      pip install pyqlib
    ```
3.  你需要准备 Qlib 数据。请遵循 [Qlib 官方指南](https://github.com/microsoft/qlib) 下载并在本地配置数据。示例脚本假设你使用的是日频数据。

### 步骤 1：配置你的实验

所有关于数据、训练和模型路径的设置都集中在 `finetune/config.py` 中。在运行任何脚本之前，请根据你的环境**修改以下路径**：

*   `qlib_data_path`：你本地 Qlib 数据目录的路径。
*   `dataset_path`：保存处理后训练/验证/测试 pickle 文件的目录。
*   `save_path`：保存模型检查点的基础目录。
*   `backtest_result_path`：保存回测结果的目录。
*   `pretrained_tokenizer_path` 与 `pretrained_predictor_path`：你希望作为起点的预训练模型路径（可以是本地路径或 Hugging Face 模型名称）。

你也可以调整其他参数，如 `instrument`、`train_time_range`、`epochs` 和 `batch_size` 以适配你的具体任务。如果你不使用 [Comet.ml](https://www.comet.com/)，请设置 `use_comet = False`。

### 步骤 2：准备数据集

运行数据预处理脚本。该脚本会从你的 Qlib 目录加载原始行情数据，进行处理，切分为训练、验证与测试集，并保存为 pickle 文件。

```shell
python finetune/qlib_data_preprocess.py
```

运行后，你将在配置中 `dataset_path` 指定的目录下找到 `train_data.pkl`、`val_data.pkl` 与 `test_data.pkl`。

### 步骤 3：运行微调

微调过程包含两个阶段：先微调 tokenizer，再微调 predictor。两个训练脚本都设计为使用 `torchrun` 进行多 GPU 训练。

#### 3.1 微调 Tokenizer

此步骤将 tokenizer 调整到你特定领域的数据分布。

```shell
# 将 NUM_GPUS 替换为你想使用的 GPU 数量（例如 2）
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_tokenizer.py
```

最佳的 tokenizer 检查点将保存到 `config.py` 中配置的路径（由 `save_path` 与 `tokenizer_save_folder_name` 派生）。

#### 3.2 微调 Predictor

此步骤针对预测任务微调 Kronos 主模型。

```shell
# 将 NUM_GPUS 替换为你想使用的 GPU 数量（例如 2）
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_predictor.py
```

最佳的 predictor 检查点将保存到 `config.py` 中配置的路径。

### 步骤 4：通过回测进行评估

最后，运行回测脚本评估你微调后的模型。该脚本会加载模型，在测试集上进行推理，生成预测信号（如预测的价格变动），并运行一个简单的 top-K 策略回测。

```shell
# 指定用于推理的 GPU
python finetune/qlib_test.py --device cuda:0
```

该脚本将在控制台输出详细的性能分析，并生成一张展示你的策略相对基准的累计收益曲线图，如下所示：

<p align="center">
    <img src="figures/backtest_result_example.png" alt="Backtest Example" align="center" width="700px" />
</p>

### 💡 从 Demo 到生产：重要注意事项

*   **原始信号 vs. 纯 Alpha**：本 Demo 中模型生成的信号是原始预测。在真实的量化工作流中，这些信号通常会被输入到投资组合优化模型中。该模型会施加约束以中性化对常见风险因子（如市场 beta、规模和价值等风格因子）的暴露，从而分离出**「纯 alpha」**并提升策略的稳健性。
*   **数据处理**：所提供的 `QlibDataset` 仅为示例。对于不同的数据源或格式，你需要调整数据加载与预处理逻辑。
*   **策略与回测复杂度**：此处使用的简单 top-K 策略只是一个基础起点。生产级策略通常包含更复杂的逻辑，用于组合构建、动态仓位管理与风险控制（如止损/止盈规则）。此外，高保真度的回测应当细致地建模交易成本、滑点和市场冲击，以便更准确地估计真实表现。

> **📝 AI 生成的注释**：请注意，`finetune/` 目录中的许多代码注释是由 AI 助手（Gemini 2.5 Pro）为解释目的而生成的。尽管它们旨在提供帮助，但可能存在不准确之处。我们建议将代码本身作为逻辑的权威依据。

## 📖 引用

如果你在研究中使用了 Kronos，欢迎引用我们的[论文](https://arxiv.org/abs/2508.02739)：

```
@misc{shi2025kronos,
      title={Kronos: A Foundation Model for the Language of Financial Markets}, 
      author={Yu Shi and Zongliang Fu and Shuo Chen and Bohan Zhao and Wei Xu and Changshui Zhang and Jian Li},
      year={2025},
      eprint={2508.02739},
      archivePrefix={arXiv},
      primaryClass={q-fin.ST},
      url={https://arxiv.org/abs/2508.02739}, 
}
```

## 📜 许可证 
本项目基于 [MIT 许可证](./LICENSE) 授权。
