"""测试脚本（与训练 / 验证分离）。

职责：仅做「测试」。加载某训练版本产出的微调 tokenizer + 预测器，在
DataSet/test 持出集（最新一段、从未参与训练 / 验证）上计算指标，结果写入
runs/<exp>/<version>/test/summary.json。

版本对齐：--version 指定训练版本；留空则自动取 runs/<exp> 下最近一次训练。

用法：
    python finetune_csv/run_test.py --config finetune_csv/configs/config_smoke20.yaml
    python finetune_csv/run_test.py --config ... --version 20260624_xxxx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (str(_THIS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline import PipelineConfig, run_eval_stage  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Kronos 测试（在 DataSet/test 上评估）")
    parser.add_argument("--config", type=str,
                        default=str(_THIS_DIR / "configs" / "config_smoke20.yaml"),
                        help="统一配置 YAML 路径")
    parser.add_argument("--version", type=str, default="",
                        help="训练版本号（留空=最近一次训练）")
    args = parser.parse_args()

    cfg = PipelineConfig(args.config)
    result = run_eval_stage(cfg, "test", version=args.version or None)
    print("测试指标：", result["metrics"])


if __name__ == "__main__":
    main()
