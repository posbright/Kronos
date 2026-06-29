"""把 dataC 里过大的单个 CSV（factors.csv / price.csv）按 symbol 拆成 N 份小文件。

动机
    DataSet/dataC/train/factors.csv 体积可达数 GB，单文件难以上传 / 管理。
    本脚本按 symbol 分桶（同一只股票永远落在同一份，不跨文件），流式逐行写出，
    内存占用极低，可处理千万级行。

拆分结果
    与原文件同目录：factors.part01.csv ... factors.part10.csv（每份带表头）。
    原文件可保留也可删除；step3（build_dataC_step3_fusion.py）会自动识别 part 文件。

用法
    # 拆分 train 的 factors.csv（默认 10 份），拆完删除原文件
    python finetune_csv/split_dataC_parts.py \
        --file DataSet/dataC/train/factors.csv --parts 10 --remove-source

    # 同时拆 price.csv
    python finetune_csv/split_dataC_parts.py --file DataSet/dataC/train/price.csv --parts 10
"""

import argparse
import csv
import sys
import zlib
from pathlib import Path


def _bucket(symbol: str, parts: int) -> int:
    """稳定哈希分桶：同一 symbol 始终落到同一份。"""
    return zlib.crc32(symbol.encode("utf-8")) % parts


def split_file(path: Path, parts: int, remove_source: bool = False) -> list:
    if parts < 2:
        raise ValueError("--parts 必须 >= 2")
    if not path.exists():
        raise FileNotFoundError(path)

    stem, suffix = path.stem, path.suffix  # factors, .csv
    out_paths = [path.with_name(f"{stem}.part{i+1:02d}{suffix}") for i in range(parts)]

    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    counts = [0] * parts
    with path.open("r", newline="", encoding="utf-8") as fin:
        reader = csv.reader(fin)
        header = next(reader)
        try:
            sym_idx = header.index("symbol")
        except ValueError as e:
            raise ValueError("CSV 缺少 'symbol' 列，无法按股票拆分") from e

        writers, handles = [], []
        for p in out_paths:
            h = p.open("w", newline="", encoding="utf-8")
            w = csv.writer(h)
            w.writerow(header)
            handles.append(h)
            writers.append(w)
        try:
            for row in reader:
                b = _bucket(row[sym_idx], parts)
                writers[b].writerow(row)
                counts[b] += 1
        finally:
            for h in handles:
                h.close()

    for p, c in zip(out_paths, counts):
        print(f"  {p.name}: {c:,} 行")
    print(f"[split] 共 {sum(counts):,} 行 -> {parts} 份")
    if remove_source:
        path.unlink()
        print(f"[split] 已删除原文件 {path.name}")
    return out_paths


def main() -> None:
    ap = argparse.ArgumentParser(description="按 symbol 把大 CSV 拆成 N 份")
    ap.add_argument("--file", required=True, help="要拆分的 CSV（含 symbol 列）")
    ap.add_argument("--parts", type=int, default=10)
    ap.add_argument("--remove-source", action="store_true", help="拆完删除原文件")
    args = ap.parse_args()
    split_file(Path(args.file), args.parts, args.remove_source)


if __name__ == "__main__":
    main()
