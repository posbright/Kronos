# A股全市场训练集构建方案（Quantia 集成）

## 1. 需求分析

目标：

- 在 Kronos 项目中增加 LightGBM 依赖。
- 基于 Quantia 的数据体系构建 A 股全市场训练数据集。
- 数据来源采用“本地缓存优先 + 远程数据库回退”的策略。
- 输出标准化 train/validation/test 三个数据分片到 `Kronos/DataSet`。
- 兼顾远程数据库低并发约束，并给后续接入 Quantia 留出扩展点。

关键约束：

- 远程数据库仅允许低并发访问，禁止大规模并行 SQL 风暴。
- cache/hist 中存在 `.corrupt` 标记和部分 pickle 兼容风险，必须具备容错回退。
- 标签构建必须避免未来信息泄漏。

---

## 2. 技术现状核查结论

已核查事实：

- Quantia kline 缓存在 `quantia/cache/hist`，按代码前三位分目录，文件名常见：

  - `{code}qfq.gzip.pickle`
  - `{code}.gzip.pickle`

- 部分文件有 `.corrupt` 标记，读取时应直接跳过。
- Quantia 已有数据库封装：`quantia.lib.database`，含重试、连接池与表存在判断。
- 可用核心数据表：

  - `cn_stock_spot`（日线快照，含 open/high/low/close/volume/deal_amount 等）
  - `cn_stock_indicators`（技术指标）
  - `cn_stock_financial`（财务数据，报告期序列）

- 当前环境对部分旧 pickle 读取可能发生 pandas 兼容异常，因此必须提供 DB 回退路径。

---

## 3. 已落地实现

### 3.1 依赖变更

- 文件：`requirements.txt`
- 新增：`lightgbm>=4.3.0`

### 3.2 新增脚本

- 文件：`finetune_csv/build_full_market_dataset.py`
- 主要能力：

  1. 自动扫描 `cache/hist` 的 A 股代码。
  2. 优先读取本地缓存（过滤 `.corrupt`）。
  3. 缓存不可用时，回退 `cn_stock_spot` 拉取 kline。
  4. 可选从 DB 叠加：
     - 技术指标（`cn_stock_indicators`）
     - 财务因子（`cn_stock_financial`）
  5. 本地补充基础时序特征（ret/ma/波动等）。
  6. 生成标签：`label_fwd_ret_{H}d`。
  7. 泄漏防护切分：
     - 训练集样本要求 `date` 与 `future_date` 均在 train 边界内。
     - 验证集样本要求 `date` 与 `future_date` 均在 validation 边界内。
     - 跨边界样本自动丢弃。
  8. 输出到：
     - `DataSet/train/dataset.csv`
     - `DataSet/validation/dataset.csv`
     - `DataSet/test/dataset.csv`
  9. 生成审计报告：`DataSet/build_report.json`。

---

## 4. 低并发数据库策略

已在脚本中执行的控制：

- 默认串行处理 symbol，避免并发连接暴涨。
- 每次 DB 查询后 sleep（`--db-sleep`，默认 0.05s）。
- 查询重试（`--db-retries`，默认 3）。
- 先检查表存在和列存在，再做字段级 SQL，避免失败重试风暴。

可继续强化（后续迭代建议）：

- 增加“每分钟最大查询数”令牌桶。
- 对 financial/indicators 增量落盘做二级缓存（本地 parquet），减少重复查询。

---

## 5. 数据切分与防泄漏设计

标签定义：

- `label_fwd_ret_Hd = close[t+H] / close[t] - 1`

切分规则（核心）：

- train: `date <= train_end` 且 `future_date <= train_end`
- validation: `train_end < date <= validation_end` 且 `future_date <= validation_end`
- test: `date > validation_end`

说明：

- 通过 `future_date` 约束确保训练样本不会引用验证/测试期价格。
- 边界穿越样本（例如 date 在 train 但 future_date 在 validation）直接 purge。

---

## 6. 运行方式

### 6.1 冒烟测试

```bash
python finetune_csv/build_full_market_dataset.py --smoke
```

### 6.2 全量构建

```bash
python finetune_csv/build_full_market_dataset.py \
  --quantia-root C:/xapproject/Quantia/Quantia \
  --cache-hist-root C:/xapproject/Quantia/Quantia/quantia/cache/hist \
  --out-root C:/xapproject/Quantia/Kronos/DataSet \
  --start-date 2017-01-01 \
  --end-date 2026-12-31 \
  --train-end 2022-12-31 \
  --validation-end 2024-12-31 \
  --label-horizon 5
```

### 6.3 受控试运行（建议先做）

```bash
python finetune_csv/build_full_market_dataset.py \
  --max-symbols 200 \
  --scan-db-symbols \
  --db-sleep 0.1
```

---

## 7. 验证与审核清单

运行后请核查：

- `build_report.json` 中 `train/validation/test` 行数均 > 0。
- 三个分片日期区间符合预期且顺序正确。
- `db_kline_fallback` 数量可解释（不应异常偏高）。
- `cache_failed` 与 `.corrupt` 文件比例大体匹配。
- 抽样检查标签：
  - 对任意样本验证 `label_fwd_ret_Hd` 与原始 close 关系一致。
- 抽样检查切分边界样本：
  - train 集样本不存在 `future_date > train_end`。

---

## 8. 与 Quantia 后续集成建议

建议下一步对接：

- 将该脚本封装为 Quantia 调度任务（类似 daily_job），支持定时增量。
- 增量策略：

  - 仅更新新增交易日，并对最近 `H+N` 天滚动重算，保证标签与 rolling 特征正确。

- 增加产物版本号：

  - `dataset_version`, `feature_schema_hash`, `source_snapshot_date`。

- 将输出格式升级为 parquet（按 split+year 分区）以提高训练吞吐。

---

## 9. 风险与缓释

风险：

- 某些历史 pickle 文件与当前 pandas 版本不兼容。
- 远程 DB 某些字段可能在不同环境列名不一致。
- 全市场全历史构建时间较长。

缓释：

- cache 读取失败自动回退 DB。
- 通过 `information_schema` 动态探测可用列并做字段白名单选择。
- 先执行 `--max-symbols` 小样本验证，再跑全量。

---

## 10. 本次交付清单

已完成：

- [x] `requirements.txt` 新增 LightGBM
- [x] 新增 `finetune_csv/build_full_market_dataset.py`
- [x] 新增本方案文档（当前文件）
- [x] 脚本支持 smoke 验证、低并发 DB、缓存优先、泄漏防护切分、审计报告输出
