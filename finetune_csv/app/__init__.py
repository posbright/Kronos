"""App 包：独立因子调参 / 重训 / 版本对比的 Flask 应用（Phase 3）。"""

from .server import create_app
from .jobs import JobManager, Job
from .registry import discover_factor_columns, list_versions, get_version
from .factor_meta import get_meta, all_meta, group_by_category, CATEGORY_INFO
from .factor_analysis import analyze_factors

__all__ = [
    "create_app",
    "JobManager",
    "Job",
    "discover_factor_columns",
    "list_versions",
    "get_version",
    "get_meta",
    "all_meta",
    "group_by_category",
    "CATEGORY_INFO",
    "analyze_factors",
]
