"""独立因子调参 App 启动器（Phase 3）。

用法：
    python finetune_csv/run_app.py --config finetune_csv/configs/config_smoke20.yaml
    # 然后浏览器打开 http://127.0.0.1:5000

说明：
    - App 复用 DataSet/{train,validation} 做因子重训，每次调权重提交一个后台任务，
      产出落到 runs/<exp>/<version>/，可在页面列出并横向对比。
    - --smoke 仅自检应用能否正常构建路由（不起服务、不训练）。
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

from app import create_app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Kronos 因子调参 / 重训 / 版本对比 App")
    parser.add_argument("--config", type=str,
                        default=str(_THIS_DIR / "configs" / "config_smoke20.yaml"),
                        help="统一训练配置 YAML（所有重训共用）")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--smoke", action="store_true",
                        help="仅自检路由构建（不起服务）")
    args = parser.parse_args()

    application = create_app(args.config)

    if args.smoke:
        rules = sorted(r.rule for r in application.url_map.iter_rules())
        client = application.test_client()
        assert client.get("/api/factors").status_code in (200, 500)  # 路由可达
        print("[smoke] run_app 通过：已注册路由")
        for r in rules:
            print("  ", r)
        return

    print(f"启动因子调参 App: http://{args.host}:{args.port}  (config={args.config})")
    application.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
