# 06. 示例与 Web UI

## 6.1 examples 目录脚本说明

[examples/](../examples/) 目录提供了多种使用场景的脚本：

| 脚本 | 说明 |
| --- | --- |
| [prediction_example.py](../examples/prediction_example.py) | **基础预测示例**：加载模型、读取 CSV、预测并用 matplotlib 绘制收盘价与成交量对比图。 |
| [prediction_batch_example.py](../examples/prediction_batch_example.py) | **批量预测示例**：演示 `predict_batch` 同时预测多组序列。 |
| [prediction_wo_vol_example.py](../examples/prediction_wo_vol_example.py) | **无成交量预测**：演示仅有 OHLC、缺少 volume/amount 时的用法。 |
| [prediction_akshare_2024-2025.py](../examples/prediction_akshare_2024-2025.py) | 结合 **akshare** 在线获取行情数据并预测。 |
| [prediction_cn_markets_day.py](../examples/prediction_cn_markets_day.py) | **A 股日线**预测示例。 |
| [prediction_new.py](../examples/prediction_new.py) | 更新版预测脚本。 |
| [prediction_new_GUI.py](../examples/prediction_new_GUI.py) | 带图形界面（GUI）的预测脚本。 |
| [get_akshare_date_2024-2025_x.py](../examples/get_akshare_date_2024-2025_x.py) / [get_date_new.py](../examples/get_date_new.py) | 数据获取辅助脚本。 |
| [run_backtest_kronos.py](../examples/run_backtest_kronos.py) | 回测脚本。 |
| [yuce/](../examples/yuce/) | 综合分析报告（JSON）与历史回测脚本 `historical_backtest.py`。 |

### 基础示例核心片段

```python
from model import Kronos, KronosTokenizer, KronosPredictor

# 1. 加载模型与 Tokenizer
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

# 2. 实例化预测器
predictor = KronosPredictor(model, tokenizer, max_context=512)

# 3. 准备数据（lookback=400, pred_len=120）
df = pd.read_csv("./data/XSHG_5min_600977.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

# 4. 预测并绘图
pred_df = predictor.predict(df=x_df, x_timestamp=x_timestamp,
                            y_timestamp=y_timestamp, pred_len=120)
plot_prediction(kline_df, pred_df)
```

> 运行示例前请确认 `./data/` 下存在对应 CSV 数据文件，且时间戳列可被 `pd.to_datetime` 解析。

## 6.2 Web UI 可视化界面

源码：[webui/app.py](../webui/app.py)，前端模板 [webui/templates/index.html](../webui/templates/index.html)。

### 功能特性

- **多格式数据支持**：CSV、Feather 等。
- **真实模型预测**：集成真实 Kronos 模型，支持多种模型尺寸（mini / small / base）。
- **预测质量可控**：可调温度（T）、核采样（top_p）、采样数（sample_count）等。
- **多设备支持**：CPU / CUDA / MPS。
- **对比分析**：预测结果与真实数据的差异统计与误差分析。
- **K 线图展示**：基于 Plotly 的专业金融图表。

### 启动方式

```bash
cd webui

# 方式一：Python 启动脚本
python run.py

# 方式二：Shell 脚本
chmod +x start.sh && ./start.sh

# 方式三：直接运行 Flask
python app.py
```

启动后访问 <http://localhost:7070>。

### 使用步骤

1. **加载数据**：从 data 目录选择金融数据文件。
2. **加载模型**：选择 Kronos 模型与计算设备。
3. **设置参数**：调整预测质量参数。
4. **选择时间窗口**：滑块选择 400+120 个数据点的时间范围。
5. **开始预测**：点击预测按钮生成结果。
6. **查看结果**：在图表与表格中查看预测结果。

### 预测质量参数建议

| 参数 | 范围 | 建议 |
| --- | --- | --- |
| 温度 T | 0.1 ~ 2.0 | 1.2 ~ 1.5 可获得更好质量 |
| 核采样 top_p | 0.1 ~ 1.0 | 0.95 ~ 1.0 考虑更多可能性 |
| 采样数 sample_count | 1 ~ 5 | 2 ~ 3 提升质量 |

### 技术架构

- **后端**：Flask + Python
- **前端**：HTML + CSS + JavaScript
- **图表**：Plotly.js
- **数据处理**：Pandas + NumPy
- **模型**：Hugging Face Transformers

### 常见问题

| 问题 | 解决方法 |
| --- | --- |
| 端口被占用 | 修改 `app.py` 中的端口号。 |
| 缺少依赖 | 运行 `pip install -r requirements.txt`。 |
| 模型加载失败 | 检查网络连接与模型 ID。 |
| 数据格式错误 | 确认列名与格式正确（须含 OHLC）。 |

> 注意：`amount` 列不用于预测，仅用于展示；时间窗口固定为 400+120=520 个数据点；首次加载模型可能需要下载，请耐心等待。

---

返回：[文档首页](./README.md)
