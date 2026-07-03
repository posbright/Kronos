# 方案 C · Linux + 8GB GPU 全年训练运行手册

> 本文是**专为 Linux GPU 服务器（8GB 显存即可）跑「全年」方案 C 全流程**整理的独立 runbook：
> 从全年特征生成 → 融合切分 → 选型 → **全量训练** → 评估 → 上线预测，**全部命令相对路径、可直接照抄**。
> 配套设计原理见 [方案C_外部融合集成.md](方案C_外部融合集成.md)，全平台逐步版见 [方案C操作指南_数据到训练验证测试.md](方案C操作指南_数据到训练验证测试.md)。
>
> 关键结论先说：
> - **唯一吃 GPU 的是步骤 2**（Kronos 衍生特征，逐窗自回归采样）；**步骤 3 融合、步骤 5 选型、步骤 6 训练、步骤 8 预测全是纯 CPU**。
> - Kronos-base/small 权重仅数百 MB，**8GB 显存绰绰有余**；显存不是瓶颈，吞吐量才是。
> - **当前仓库里的 `DataSet/dataC` 不是 180+180 的全年窗**：`validation+test` 每股最多约 239 行，因此 step2 里 `--recent-days 250` 会报历史长度不足；按当前数据应改用 `--recent-days 120` 左右（`lookback=90,pred=5` 时上限约为 144）。若你先用 step1 重新生成更长的 `dataC`，再把 `--recent-days` 提回 250。

---

## 0. 适用前提

| 项 | 要求 |
| --- | --- |
| 系统 | Linux（Ubuntu/CentOS 等） |
| GPU | NVIDIA，显存 ≥ 8GB；`nvidia-smi` 正常 |
| 显存占用 | Kronos-base ≈ 几百 MB，8GB 富余 |
| 当前 dataC | `validation+test` 每股最多约 239 行；step2 建议 `--recent-days 120`，`lookback=90,pred=5` 时上限约 144 |
| 全量训练 | C1 下游模型（LightGBM/Ridge）**纯 CPU**，无需 GPU |

> **"全量"含义**：本手册指**尽量多标的 + 尽量长的可用窗口**。在当前仓库 `dataC` 上，`validation+test` 只能支撑最近窗口推理，
> 建议先用数百只标的跑通 `--recent-days 120`；若你重新构建了更长的 `dataC`，再把 `recent-days` 提回 250。要全市场需多卡或改 `predict_batch` 批并行（见 4.3）。

---

## 1. 一次性环境准备

### 1.1 取代码 + 建 GPU 版 venv

```bash
git clone https://github.com/posbright/Kronos.git
cd Kronos

python3 -m venv .venv
source .venv/bin/activate
# 查看CUDA的版本
nvidia-smi
# 安装命令后添加 -i 参数指定国内镜像源，例如使用清华大学的源：
pip install sympy -i https://pypi.tuna.tsinghua.edu.cn/simple
# 阿里云的源：
pip install sympy -i https://mirrors.aliyun.com/pypi/simple/
# 失败处理方式
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# CUDA 版 torch（按服务器 CUDA 版本选 index-url，cu121 示例）
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt   # 已含 lightgbm/pyyaml/einops/tqdm 等全部依赖

python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> **关键坑**：CPU 版 `.venv` 直接拷到 GPU 机仍只用 CPU。必须在 GPU 机上重建 venv 并装 **CUDA 版 torch**。

### 1.2 数据就位（`DataSet/dataC` 已 gitignore，不随 git 走）

二选一：

- **重建（能连 Quantia DB/cache 时推荐）**：在本机执行步骤 1（见全平台指南第 2 节），产出 `DataSet/dataC/{train,validation,test}/`。
- **拷贝**：把 `DataSet/dataC` 传到 GPU 机相同相对路径。大文件可先拆分（见 5.1），再 `rsync`：

```bash
rsync -avz user@srchost:/path/Kronos/DataSet/dataC/ ./DataSet/dataC/
```

### 1.3 预训练权重

默认 **ModelScope 优先、HF 兜底**（`AI-ModelScope/*` → `NeoQuasar/*`）。离线机预置 `~/.cache/modelscope`、`~/.cache/huggingface`，或本地权重目录传 `--tokenizer/--predictor`。

### 1.4 当前数据下的耗时估算（8GB 单卡）

成本 = 标的数 × 窗口数 × samples × 单次耗时；GPU 单次 ≈ 0.05s。窗口数 ≈ `recent-days+1`。
当前仓库 `dataC` 的 `validation+test` 最多只够 `recent-days≈120` 级别的演示窗口；下表按这个更稳妥的档位估算。若你重新生成了更长数据，再按 250 天近似线性放大。

| 规模 | 标的 | recent-days | samples | 估算 |
| --- | --- | --- | --- | --- |
| 验证 | 50 | 120 | 30 | ≈ 2.4 h |
| 推荐 | 300 | 120 | 30 | ≈ 14.9 h |
| 全市场（仅在重建更长 dataC 后） | 6000 | 250 | 30 | ≈ 26 天（需 `predict_batch`/多卡；`--max-symbols 0` 一次全量） |

---

## 2. 步骤 2：GPU 生成全年 Kronos 特征（唯一耗 GPU）

```bash
source .venv/bin/activate

python finetune_csv/build_dataC_step2_kronos_features.py \
    --data-root DataSet/dataC \
    --device cuda:0 \
    --max-symbols 300 --recent-days 120 \
    --lookback 90 --pred 5 --samples 30 --seed 42 \
    --skip-existing
```

- `--device cuda:0`：显式 GPU；无 CUDA 自动回退 CPU 并告警。
- `--recent-days 120` 适配当前仓库 `dataC`；如果你先按 step1 重建出更长的 `validation/test`，再把它提回 250。
- `--samples 30` 生产建议 ≥30。
- `--skip-existing`（默认开）：逐只增量落 `DataSet/dataC/_kronos_parts/`，**中断可续跑**，重跑不丢已完成标的。
- **产物**：`DataSet/dataC/kronos_features.csv`（`date,symbol,k_pred_ret,k_up_prob,k_pred_vol`）+ `kronos_features_report.json`（含 `device_resolved`）。
- 后台长跑：`nohup python finetune_csv/build_dataC_step2_kronos_features_batch.py ... > step2_full.log 2>&1 &`，或用 `tmux`/`screen`。
- 日志会实时刷出总进度百分比、单只标的 batch 进度、平均耗时、速度和 ETA；用 `tail -f step2_full.log` 观察即可。

上面是 **300 只示例**（先跑通）。**要跑当前窗口内的全部合格标的**，把 `--max-symbols 300` 改成 `--max-symbols 0` 即可（建议同时用批并行版 + 后台长跑）；这不等于全年历史，只是把当前 dataC 里满足历史长度要求的标的全跑一遍。

```bash
nohup python finetune_csv/build_dataC_step2_kronos_features_batch.py \
    --data-root DataSet/dataC --device cuda:0 \
    --max-symbols 0 --recent-days 120 \
    --lookback 90 --pred 5 --samples 30 --seed 42 \
    --batch-size 512 --symbol-group-size 4 \
    --skip-existing > step2_full.log 2>&1 &
```

> 步骤 2 只读取 `validation+test`。**当前仓库 dataC 不是全年窗**，因此 `--recent-days 250` 会失败；先用 120 跑通，或先重建更长数据后再恢复 250。24G A10 建议从 `--batch-size 512 --symbol-group-size 4` 起步；稳定后可试 `--symbol-group-size 8` 或 `--batch-size 768`。8GB 显存仍建议 `--batch-size 192 --symbol-group-size 1`。

### 2.1 24G A10 上的优化版完整脚本（直接复制）

适用于你当前的 NVIDIA A10 24G。该脚本只加速 step2 的 Kronos 特征生成，不改变最终特征含义；已有 `_kronos_parts/*.csv` 会被 `--skip-existing` 自动复用。

```bash
cd /mnt/workspace/Kronos_back/Kronos
source .venv/bin/activate

# 1) 如果之前旧版 step2 还在跑，先停掉；没有进程则跳过。
ps -ef | grep -E "build_dataC_step2|python" | grep -v grep
# kill <旧step2进程PID>

# 2) 拉取包含 --symbol-group-size 的新版代码。
git pull

# 3) 启动 24G A10 优化版 step2。
nohup python finetune_csv/build_dataC_step2_kronos_features_batch.py \
    --data-root DataSet/dataC --device cuda:0 \
    --max-symbols 0 --recent-days 120 \
    --lookback 90 --pred 5 --samples 30 --seed 42 \
    --batch-size 512 --symbol-group-size 4 \
    --skip-existing > step2_full.log 2>&1 &

# 4) 观察进度、显存和最近速度。
tail -f step2_full.log
```

另开一个终端可随时查看进度和 ETA：

```bash
done=$(ls DataSet/dataC/_kronos_parts/*.csv 2>/dev/null | wc -l)
recent30=$(find DataSet/dataC/_kronos_parts -name "*.csv" -mmin -30 | wc -l)
echo "done=$done, remain=$((5723-done)), recent30=$recent30, rate_per_hour=$((recent30*2))"
nvidia-smi
```

若 `step2_full.log` 稳定推进、显存低于 80%，可下次尝试 `--symbol-group-size 8` 或 `--batch-size 768`。若 OOM，优先回退到 `--symbol-group-size 4`，再把 `--batch-size` 降到 384。

step2 完成后再执行后续 CPU 步骤：

```bash
python finetune_csv/build_dataC_step3_fusion.py \
    --data-root DataSet/dataC --horizon 5 --train-end 2026-04-01 --val-end 2026-05-15

python finetune_csv/compare_fusion_strategies.py \
    --train DataSet/dataC/fusion_train.csv --val DataSet/dataC/fusion_val.csv \
    --test DataSet/dataC/fusion_test.csv --label label_fwd_ret_5d \
    --out-json DataSet/dataC/fusion_selection.json

python finetune_csv/train_c1_bundle.py \
    --data-root DataSet/dataC --out-bundle runs/dataC_c1 --horizon 5

python finetune_csv/train_c1_bundle.py --predict \
    --data-root DataSet/dataC --out-bundle runs/dataC_c1 \
    --top 10 --out-json runs/dataC_c1/latest_ranking.json
```

### 2.2 `lookback` / `pred` 能否增大

可以适量增大，但受当前数据长度硬约束：

```text
recent-days + lookback + pred <= 每股可用行数
```

当前 `validation+test` 每股最多约 239 行；默认 `120 + 90 + 5 = 215`，仍有约 24 行余量。比较稳妥的组合如下：

| 目标 | recent-days | lookback | pred | 最少行数 | 说明 |
| --- | --- | --- | --- | --- | --- |
| 默认稳妥 | 120 | 90 | 5 | 215 | 当前推荐；覆盖标的最多，速度和稳定性平衡 |
| 增大回看 | 120 | 100 | 5 | 225 | 可用；多看历史，预测周期不变 |
| 小幅增大预测 | 120 | 100 | 10 | 230 | 可用；建议同步把 step3/训练标签改成 10 天 |
| 更长预测 | 120 | 90 | 20 | 230 | 可用但波动更大；同样要同步 `--horizon 20` 与标签 |
| 不建议 | 120 | 120 | 20 | 260 | 当前 dataC 会筛不到标的 |

原则：优先小幅提高 `lookback` 到 100 或把 `pred` 提到 10；不要同时大幅提高两者。若 `pred` 从 5 改成 10/20，后续 step3、compare、train、predict 的 `--horizon`/`label_fwd_ret_{H}d` 也要同步改成同一个 H，否则 Kronos 特征预测的是 10/20 天，但下游模型训练的仍是 5 天收益，目标会错位。

### 2.3 为什么默认 300？全市场 6000 只怎么训完

`--max-symbols 300` 只是**示例规模**——300 只单卡在当前 dataC 上按 `recent-days=120` 约十几个小时量级，便于先跑通；如果你重建出更长 dataC，再按 250 天近似放大。覆盖全量两种方式（**先 step2 跑完当前窗口内全部合格标的，step3/5/6 再统一训练一个模型**，并非每批单独训一个）：

**方式 A：一次全量（推荐，最省心）** —— `--max-symbols 0` 即“当前窗口内全部合格标的”，`--skip-existing` 逐只落 part，**中断随时续跑**：

```bash
python finetune_csv/build_dataC_step2_kronos_features_batch.py \
    --device cuda:0 --max-symbols 0 --recent-days 120 \
    --samples 30 --batch-size 512 --symbol-group-size 4 --skip-existing
```

**方式 B：分 300 一批、可多机/多窗口并行** —— 用 `--symbol-offset` 切片，**同一 `--seed` 下批次互不重叠**，跑完拼成一份 `kronos_features.csv`：

```bash
for off in 0 300 600 900 ... 5700; do
  python finetune_csv/build_dataC_step2_kronos_features_batch.py \
      --device cuda:0 --seed 42 --symbol-offset $off --max-symbols 300 \
            --recent-days 120 --samples 30 \
            --batch-size 512 --symbol-group-size 4 --skip-existing
done
```

> 关键：**当前仓库 dataC 共用一个 step2 输出 + 一个 step3/6 模型**，不是每批单独训一个。part 写在 `_kronos_parts/{symbol}.csv`，无论分多少批，最后 step2 都汇总成同一张 `kronos_features.csv` 供后续步骤一次性训练。多机时各机跑不同 offset，汇总 part 目录即可。

---

## 3. 步骤 3：融合宽表 + 时间切分（CPU）

```bash
python finetune_csv/build_dataC_step3_fusion.py \
    --data-root DataSet/dataC --horizon 5 \
    --train-end 2026-04-01 --val-end 2026-05-15
```

- 纯 CPU、向量化，全年数据**无需改动**；默认从 `validation,test` 合并 `factors/price`。
- **显式切分边界**（`--train-end/--val-end`）用于复现；留空则按交易日 70/15/15 自动分位。
- 自动识别 `factors.csv` 或拆分的 `factors.part*.csv`（见 5.1）。
- 产物：`DataSet/dataC/fusion_{all,train,val,test}.csv` + `fusion_report.json`。

---

## 4. 步骤 5/6：选型 + 全量训练（CPU）

### 4.1 验证集选型（C1 主线 / C2 兜底）

```bash
python finetune_csv/compare_fusion_strategies.py \
    --train DataSet/dataC/fusion_train.csv --val DataSet/dataC/fusion_val.csv \
    --test DataSet/dataC/fusion_test.csv --label label_fwd_ret_5d \
    --switch-threshold 0.005 --out-json DataSet/dataC/fusion_selection.json
```

### 4.2 全量训练 C1 bundle（多标的主线，**纯 CPU**）

```bash
python finetune_csv/train_c1_bundle.py \
    --data-root DataSet/dataC --out-bundle runs/dataC_c1 --horizon 5 --backend auto
```

- 自动找 `fusion_{train,val,test}.csv`；特征列自动识别（kronos 列在前）。
- `--backend auto`：LightGBM 优先，未装回退 Ridge；**训练不用 GPU**。
- 产物 bundle：`runs/dataC_c1/`（`manifest.json` + `c1_lgb.txt`/`c1_ridge.npz`），与 `run_fusion.py` 互通。

### 4.3 全市场提速（可选，predict_batch 批并行）

单卡逐窗串行偏慢；用批并行版脚本把「多窗 × 多采样」打包成一个 batch 一次前向。`--symbol-group-size` 可以进一步把多只标的一起合并进 `predict_batch`，减少单只标的之间的 Python/GPU 空档；只要 `lookback/pred/samples/seed` 不变，产物列和含义不变，仍按 symbol 写回 `_kronos_parts/{symbol}.csv`。

```bash
python finetune_csv/build_dataC_step2_kronos_features_batch.py \
    --device cuda:0 --max-symbols 300 --recent-days 120 \
    --lookback 90 --pred 5 --samples 30 \
    --batch-size 512 --symbol-group-size 4 --skip-existing
```

`--batch-size` = 一次进显存的并行序列数（即 窗口×samples）；`--symbol-group-size` = 一组同时准备多少只标的。OOM 就优先把 `--symbol-group-size` 减半，再降 `--batch-size`。底层走 `KronosPredictor.predict_batch`（[model/kronos.py](../../model/kronos.py)），同批 `lookback`/`pred` 须一致（脚本已保证）。

| 显存 | 起步参数 | 可上探 | 说明 |
| --- | --- | --- | --- |
| 6G | `--batch-size 96 --symbol-group-size 1` | batch 128 | OOM 先降到 64；样本/序列长越大越要降 |
| 8G | `--batch-size 192 --symbol-group-size 1` | batch 256 | 8GB 不建议跨 symbol 合并，优先稳 |
| 24G A10 | `--batch-size 512 --symbol-group-size 4` | group 8 / batch 768 | 若显存长期低于 80% 且日志稳定推进，再逐步上探；OOM 回退 group 4 / batch 384 |

相对逐窗版吞吐通常提升 3~8 倍；6G 用 96、8G 用 192 是稳妥默认。

---

## 5. 大文件与上线

### 5.1 大文件拆分（迁移友好）

```bash
python finetune_csv/split_dataC_parts.py --file DataSet/dataC/train/factors.csv --parts 10
python finetune_csv/split_dataC_parts.py --file DataSet/dataC/train/price.csv  --parts 5
```

按 symbol 拆 N 份（同股不跨份），step3 自动识别单文件或 `part` 分片，命令不变。

### 5.2 上线截面打分（CPU）

```bash
python finetune_csv/train_c1_bundle.py --predict \
    --data-root DataSet/dataC --out-bundle runs/dataC_c1 \
    --top 10 --out-json runs/dataC_c1/latest_ranking.json
```

---

## 6. 设备/平台校验清单

- [ ] `python -c "import torch;print(torch.cuda.is_available())"` → True。
- [ ] 步骤 2 启动日志打印 `device_resolved=cuda:0`。
- [ ] 步骤 3/5/6/8 CPU 运行，无需 GPU。
- [ ] 文件名大小写照抄（`DataSet` ≠ `dataset`）；CRLF 脚本用 `dos2unix` 修。
- [ ] 长任务用 `nohup/tmux`；step2 断点续跑可恢复。

---

## 7. 全年一键串联（GPU step2 + CPU 其余）

```bash
source .venv/bin/activate

python finetune_csv/build_dataC_step2_kronos_features.py \
    --data-root DataSet/dataC --device cuda:0 \
    --max-symbols 300 --recent-days 120 --samples 30 --skip-existing

python finetune_csv/build_dataC_step3_fusion.py \
    --data-root DataSet/dataC --horizon 5 --train-end 2026-04-01 --val-end 2026-05-15

python finetune_csv/compare_fusion_strategies.py \
    --train DataSet/dataC/fusion_train.csv --val DataSet/dataC/fusion_val.csv \
    --test DataSet/dataC/fusion_test.csv --label label_fwd_ret_5d \
    --out-json DataSet/dataC/fusion_selection.json

python finetune_csv/train_c1_bundle.py \
    --data-root DataSet/dataC --out-bundle runs/dataC_c1 --horizon 5

python finetune_csv/train_c1_bundle.py --predict \
    --data-root DataSet/dataC --out-bundle runs/dataC_c1 \
    --top 10 --out-json runs/dataC_c1/latest_ranking.json
```

### 7.1 全市场全量一键串联（仅 step2 改 `--max-symbols 0`，其余不变）

```bash
source .venv/bin/activate

# step2：全市场 6000 只全量（批并行 + 后台续跑）
nohup python finetune_csv/build_dataC_step2_kronos_features_batch.py \
    --data-root DataSet/dataC --device cuda:0 \
    --max-symbols 0 --recent-days 120 --samples 30 \
    --batch-size 512 --symbol-group-size 4 \
    --skip-existing > step2_full.log 2>&1 &
wait

# step3/5/6/8：与上面 300 只完全一致，一份特征训一个模型
python finetune_csv/build_dataC_step3_fusion.py \
    --data-root DataSet/dataC --horizon 5 --train-end 2026-04-01 --val-end 2026-05-15
python finetune_csv/compare_fusion_strategies.py \
    --train DataSet/dataC/fusion_train.csv --val DataSet/dataC/fusion_val.csv \
    --test DataSet/dataC/fusion_test.csv --label label_fwd_ret_5d \
    --out-json DataSet/dataC/fusion_selection.json
python finetune_csv/train_c1_bundle.py \
    --data-root DataSet/dataC --out-bundle runs/dataC_c1 --horizon 5
python finetune_csv/train_c1_bundle.py --predict \
    --data-root DataSet/dataC --out-bundle runs/dataC_c1 \
    --top 10 --out-json runs/dataC_c1/latest_ranking.json
```

> 当前 `recent-days=120` 时，全市场单卡 step2 约十几天量级（见 1.4）；若重建更长 dataC 并提回 250 天，约 26 天。建议多机分 `--symbol-offset` 并行（见 2.3 方式 B），或缩小 `--recent-days`/`--samples`。step3/5/6/8 仍是几分钟级 CPU。
