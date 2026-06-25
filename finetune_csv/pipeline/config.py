"""统一流水线配置 PipelineConfig。

目标：所有 run_*.py（构建数据 / 训练 / 验证 / 测试）共用同一份 YAML，避免参数散落各处、
难以维护。本类在原有 config_loader.ConfigLoader 之上，新增数据源、股票池、滚动切分、因子、
日志/版本等小范围验证流水线所需的配置段，并提供带默认值的类型化读取接口。

YAML 结构（新增段，旧段沿用 config_loader）：

    data_source:            # 数据来源（Quantia 缓存 + 远程 DB）
      quantia_root: "C:/xapproject/Quantia/Quantia"
      cache_hist_root: ""   # 留空则自动 = quantia_root/quantia/cache/hist
      start_date: "2017-01-01"
      end_date: ""          # 留空 = 今天
      db_sleep: 0.1         # 每次 DB 查询后 sleep 秒数（低并发保护）
      db_retries: 3
      use_db_features: true # 是否用 DB 补充技术指标 / 基本面

    universe:               # 股票池
      symbols: []           # 显式股票列表；为空则自动从缓存挑选
      max_symbols: 20       # 小范围验证默认 20 只
      min_history: 250      # 最少交易日（过滤新股 / 数据过短）

    splits:                 # 滚动日期切分（详见 splits.py）
      anchor: ""            # 锚点；留空 = 今天-1
      test_days: 40
      val_days: 40
      train_days: 0         # 0 / null = 锚点之前剩余全部
      label_horizon: 5      # 前看收益标签步数

    factors:                # 因子缺失处理（详见 factors.py）
      add_mask: true
      mask_threshold: 0.05
      strategy: {}          # 显式覆盖 {列名: ffill|zero}

    runs:                   # 输出与版本
      runs_root: ""         # 留空 = <repo>/finetune_csv/runs
      version: ""           # 留空 = 时间戳
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from config_loader import ConfigLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FINETUNE_CSV = Path(__file__).resolve().parents[1]


class PipelineConfig:
    """统一配置封装。用 cfg.get('a.b.c', default) 取任意值，或用类型化属性取常用值。"""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.loader = ConfigLoader(config_path)
        # 版本号在进程内只解析一次，确保同一次运行里 build/train 各阶段写入同一目录。
        self._version: Optional[str] = None

    # ---- 通用读取 ----
    def get(self, key: str, default: Any = None) -> Any:
        return self.loader.get(key, default)

    # ---- 数据来源 ----
    @property
    def quantia_root(self) -> Path:
        return Path(self.get("data_source.quantia_root", "C:/xapproject/Quantia/Quantia"))

    @property
    def cache_hist_root(self) -> Path:
        raw = self.get("data_source.cache_hist_root", "")
        if raw:
            return Path(raw)
        return self.quantia_root / "quantia" / "cache" / "hist"

    @property
    def start_date(self) -> str:
        return str(self.get("data_source.start_date", "2017-01-01"))

    @property
    def end_date(self) -> str:
        raw = self.get("data_source.end_date", "")
        return str(raw) if raw else str(dt.date.today())

    @property
    def db_sleep(self) -> float:
        return float(self.get("data_source.db_sleep", 0.1))

    @property
    def db_retries(self) -> int:
        return int(self.get("data_source.db_retries", 3))

    @property
    def use_db_features(self) -> bool:
        return bool(self.get("data_source.use_db_features", True))

    @property
    def tech_source(self) -> str:
        """技术指标来源：'local'=本地用 OHLCV 现算（默认，历史完整）；'db'=远程库 cn_stock_indicators。"""
        return str(self.get("data_source.tech_source", "local")).lower()

    # ---- 股票池 ----
    @property
    def universe_symbols(self) -> List[str]:
        syms = self.get("universe.symbols", []) or []
        return [str(s).zfill(6) for s in syms]

    @property
    def max_symbols(self) -> int:
        return int(self.get("universe.max_symbols", 20))

    @property
    def min_history(self) -> int:
        return int(self.get("universe.min_history", 250))

    # ---- 滚动切分 ----
    @property
    def anchor(self) -> Optional[str]:
        raw = self.get("splits.anchor", "")
        return str(raw) if raw else None

    @property
    def test_days(self) -> int:
        return int(self.get("splits.test_days", 40))

    @property
    def val_days(self) -> int:
        return int(self.get("splits.val_days", 40))

    @property
    def train_days(self) -> Optional[int]:
        raw = self.get("splits.train_days", 0)
        v = int(raw) if raw else 0
        return v if v > 0 else None

    @property
    def label_horizon(self) -> int:
        return int(self.get("splits.label_horizon", 5))

    # ---- 因子缺失处理 ----
    @property
    def factor_add_mask(self) -> bool:
        return bool(self.get("factors.add_mask", True))

    @property
    def factor_mask_threshold(self) -> float:
        return float(self.get("factors.mask_threshold", 0.05))

    @property
    def factor_strategy(self) -> Dict[str, str]:
        return dict(self.get("factors.strategy", {}) or {})

    # ---- 数据集 / 训练（沿用旧段，给默认值） ----
    @property
    def lookback_window(self) -> int:
        return int(self.get("data.lookback_window", 90))

    @property
    def predict_window(self) -> int:
        return int(self.get("data.predict_window", 10))

    @property
    def max_context(self) -> int:
        return int(self.get("data.max_context", 512))

    @property
    def clip(self) -> float:
        return float(self.get("data.clip", 5.0))

    @property
    def exp_name(self) -> str:
        return str(self.get("model_paths.exp_name", "kronos_smoke20"))

    # ---- 训练超参（training 段，给默认值） ----
    @property
    def tokenizer_epochs(self) -> int:
        return int(self.get("training.tokenizer_epochs", 10))

    @property
    def basemodel_epochs(self) -> int:
        return int(self.get("training.basemodel_epochs", 10))

    @property
    def batch_size(self) -> int:
        return int(self.get("training.batch_size", 16))

    @property
    def log_interval(self) -> int:
        return int(self.get("training.log_interval", 20))

    @property
    def num_workers(self) -> int:
        return int(self.get("training.num_workers", 0))

    @property
    def seed(self) -> int:
        return int(self.get("training.seed", 42))

    @property
    def tokenizer_learning_rate(self) -> float:
        return float(self.get("training.tokenizer_learning_rate", 2e-4))

    @property
    def predictor_learning_rate(self) -> float:
        return float(self.get("training.predictor_learning_rate", 1e-5))

    @property
    def adam_beta1(self) -> float:
        return float(self.get("training.adam_beta1", 0.9))

    @property
    def adam_beta2(self) -> float:
        return float(self.get("training.adam_beta2", 0.95))

    @property
    def adam_weight_decay(self) -> float:
        return float(self.get("training.adam_weight_decay", 0.1))

    @property
    def accumulation_steps(self) -> int:
        return int(self.get("training.accumulation_steps", 1))

    @property
    def patience(self) -> int:
        """早停耐心：连续多少个 epoch 验证无提升即停。"""
        return int(self.get("training.patience", 8))

    # ---- 预训练 / 设备 ----
    @property
    def pretrained_tokenizer(self) -> str:
        return str(self.get("model_paths.pretrained_tokenizer", "NeoQuasar/Kronos-Tokenizer-base"))

    @property
    def pretrained_predictor(self) -> str:
        return str(self.get("model_paths.pretrained_predictor", "NeoQuasar/Kronos-small"))

    @property
    def use_cuda(self) -> bool:
        return bool(self.get("device.use_cuda", True))

    @property
    def device_id(self) -> int:
        return int(self.get("device.device_id", 0))

    # ---- 输出 / 版本 ----
    @property
    def runs_root(self) -> Path:
        raw = self.get("runs.runs_root", "")
        return Path(raw) if raw else (_FINETUNE_CSV / "runs")

    @property
    def version(self) -> str:
        """运行版本号。进程内缓存，避免每次访问生成不同时间戳导致阶段目录错位。

        优先级：显式 set_version() > 配置 runs.version > 运行时时间戳。
        """
        if self._version is None:
            raw = self.get("runs.version", "")
            self._version = str(raw) if raw else dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        return self._version

    def set_version(self, version: str) -> None:
        """显式指定版本号（如 run_validate/run_test 对齐 run_train 产出的版本）。"""
        self._version = str(version)

    def latest_version(self) -> Optional[str]:
        """返回 runs/<exp_name> 下最近一次（按目录名排序）的版本号；无则 None。

        用于 run_validate/run_test 在未显式指定 --version 时自动定位最新一次训练。
        """
        exp_dir = self.runs_root / self.exp_name
        if not exp_dir.is_dir():
            return None
        versions = sorted(p.name for p in exp_dir.iterdir() if p.is_dir())
        return versions[-1] if versions else None

    @property
    def dataset_root(self) -> Path:
        raw = self.get("data.out_root", "")
        return Path(raw) if raw else (_REPO_ROOT / "DataSet")

    def run_dir(self, stage: str) -> Path:
        """返回 runs/<exp_name>/<version>/<stage> 目录（自动创建）。

        stage 例：'build' / 'train' / 'validate' / 'test'。
        """
        d = self.runs_root / self.exp_name / self.version / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.loader.config)
