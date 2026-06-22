# ============================================================================
# 示例：通过 ModelScope（魔搭社区）加载 Kronos 模型进行预测（方案一：混合方案）
# ----------------------------------------------------------------------------
# 适用场景：国内网络无法稳定访问 Hugging Face 时，改用 ModelScope 下载权重。
# 实现原理：
#   1) KronosTokenizer.from_modelscope / Kronos.from_modelscope 内部先调用
#      modelscope.snapshot_download 将模型仓库下载到本地缓存目录；
#   2) 再复用 PyTorchModelHubMixin 的 from_pretrained 从“本地目录”加载权重，
#      整个过程不会访问 Hugging Face。
#
# 依赖：pip install modelscope
# ============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor


def plot_prediction(kline_df, pred_df):
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Ground Truth'
    sr_pred_close.name = "Prediction"

    sr_volume = kline_df['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = 'Ground Truth'
    sr_pred_volume.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(close_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax1.plot(close_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax1.set_ylabel('Close Price', fontsize=14)
    ax1.legend(loc='lower left', fontsize=12)
    ax1.grid(True)

    ax2.plot(volume_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax2.plot(volume_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax2.set_ylabel('Volume', fontsize=14)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# 1. 从 ModelScope 加载模型与分词器（无需访问 Hugging Face）
tokenizer = KronosTokenizer.from_modelscope("AI-ModelScope/Kronos-Tokenizer-base")
model = Kronos.from_modelscope("AI-ModelScope/Kronos-base")

# 2. 实例化预测器
predictor = KronosPredictor(model, tokenizer, max_context=512)

# 3. 准备数据
df = pd.read_csv("./data/XSHG_5min_600977.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

lookback = 400
pred_len = 120

x_df = df.loc[:lookback - 1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = df.loc[:lookback - 1, 'timestamps']
y_timestamp = df.loc[lookback:lookback + pred_len - 1, 'timestamps']

# 4. 执行预测
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True
)

print("Forecasted Data Head:")
print(pred_df.head())

# 5. 可视化
kline_df = df.loc[:lookback + pred_len - 1]
plot_prediction(kline_df, pred_df)
