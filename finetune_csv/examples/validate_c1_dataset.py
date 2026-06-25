"""方案 C1 数据链路真实端到端校验脚本。

目的：用**真实日线数据 + 缓存的 Kronos 权重**跑通「生成训练/验证/测试集」的全过程，
      并对产出的融合数据做严格的准确性断言（标签正确性、无泄漏、无 NaN、时间切分守恒）。

数据来源（均为仓库内真实文件，非合成）：
  - 价量：examples/data/300308_stock_data.csv（A 股日线，取一段正价子集）
  - 因子：直接复用该 CSV 自带的当日可得列（turnover/pct_chg/amplitude）作为真实因子
  - 权重：本机 HuggingFace 缓存中的 NeoQuasar/Kronos-Tokenizer-base + Kronos-base

用法：
    .venv/Scripts/python.exe finetune_csv/examples/validate_c1_dataset.py
"""

import os
import sys

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FT_DIR = os.path.dirname(_THIS_DIR)              # finetune_csv/
_REPO_ROOT = os.path.dirname(_FT_DIR)             # repo root
for p in (_FT_DIR, _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from build_kronos_features import build_features          # noqa: E402
from build_fusion_dataset import build_fusion, time_split  # noqa: E402
from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402

SYMBOL = "300308"
SRC_CSV = os.path.join(_REPO_ROOT, "examples", "data", "300308_stock_data.csv")
OUT_DIR = os.path.join(_THIS_DIR, "_c1_demo")
# 取一段“价格为正、跨度足够切分”的子集（2017-11 起约 360 个交易日）
SUBSET_START, SUBSET_LEN = 1200, 360
LOOKBACK, PRED, SAMPLES, HORIZON = 64, 5, 3, 5


def load_price_subset() -> pd.DataFrame:
    raw = pd.read_csv(SRC_CSV)
    sub = raw.iloc[SUBSET_START:SUBSET_START + SUBSET_LEN].copy()
    px = pd.DataFrame({
        "timestamps": pd.to_datetime(sub["timestamps"]),
        "open": sub["open"].astype(float),
        "high": sub["high"].astype(float),
        "low": sub["low"].astype(float),
        "close": sub["close"].astype(float),
        "volume": sub["volume"].astype(float),
        "amount": sub["amount"].astype(float),
        # 当日可得的真实因子（收盘时即可知，作为下游因子特征）
        "turnover": sub["turnover"].astype(float),
        "pct_chg": sub["pct_chg"].astype(float),
        "amplitude": sub["amplitude"].astype(float),
    }).reset_index(drop=True)
    assert (px["close"] > 0).all(), "子集中存在非正收盘价，请换区间"
    return px


def main() -> None:
    print("=" * 72)
    print("方案 C1 数据链路真实端到端校验")
    print("=" * 72)

    px = load_price_subset()
    print(f"[1] 价量子集：{len(px)} 行，{px['timestamps'].min().date()} ~ "
          f"{px['timestamps'].max().date()}")

    print("[2] 加载缓存的 Kronos-base 并生成衍生特征（真实推理，CPU）...")
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    mdl = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(mdl, tok, device="cpu", max_context=512)
    kf = build_features(predictor, px[["timestamps", "open", "high", "low",
                                       "close", "volume", "amount"]],
                        symbol=SYMBOL, lookback=LOOKBACK, pred_len=PRED,
                        samples=SAMPLES, verbose=True)
    print(f"    -> Kronos 特征 {len(kf)} 行；列 = {list(kf.columns)}")

    # 真实因子表：复用价量 CSV 自带的当日可得列
    ff = px[["timestamps", "turnover", "pct_chg", "amplitude"]].copy()
    ff = ff.rename(columns={"timestamps": "date"})
    ff["symbol"] = SYMBOL

    print("[3] 对齐 Kronos 特征 + 真实因子 + 标签，并按时间切分 ...")
    fusion = build_fusion(kf, ff, px[["timestamps", "close"]].assign(symbol=SYMBOL),
                          horizon=HORIZON)
    # 按特征日期的 60% / 80% 分位动态确定切分点，保证三段非空
    uniq_dates = np.sort(fusion["date"].unique())
    train_end = pd.Timestamp(uniq_dates[int(len(uniq_dates) * 0.6)])
    val_end = pd.Timestamp(uniq_dates[int(len(uniq_dates) * 0.8)])
    tr, va, te = time_split(fusion, train_end=str(train_end.date()),
                            val_end=str(val_end.date()))

    os.makedirs(OUT_DIR, exist_ok=True)
    fusion.to_csv(os.path.join(OUT_DIR, "fusion_all.csv"), index=False)
    tr.to_csv(os.path.join(OUT_DIR, "fusion_train.csv"), index=False)
    va.to_csv(os.path.join(OUT_DIR, "fusion_val.csv"), index=False)
    te.to_csv(os.path.join(OUT_DIR, "fusion_test.csv"), index=False)
    label_col = f"label_fwd_ret_{HORIZON}d"
    feat_cols = ["k_pred_ret", "k_up_prob", "k_pred_vol",
                 "turnover", "pct_chg", "amplitude"]
    print(f"    -> 融合 {len(fusion)} 行，切分 train/val/test = "
          f"{len(tr)}/{len(va)}/{len(te)}，已写入 {OUT_DIR}")

    # ================= 准确性校验 =================
    print("\n" + "-" * 72)
    print("数据准确性校验")
    print("-" * 72)
    checks = []

    def check(name, ok, detail=""):
        checks.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    # (1) 列完整
    need = set(feat_cols + [label_col, "date", "symbol"])
    check("列完整（Kronos特征+因子+标签+索引）", need.issubset(fusion.columns),
          f"缺失={need - set(fusion.columns)}")

    # (2) 特征无 NaN
    n_nan = int(fusion[feat_cols].isnull().sum().sum())
    check("特征无 NaN", n_nan == 0, f"NaN 数={n_nan}")

    # (3) Kronos 特征取值合理
    finite = np.isfinite(fusion[["k_pred_ret", "k_up_prob", "k_pred_vol"]].values).all()
    up_ok = fusion["k_up_prob"].between(0, 1).all()
    vol_ok = (fusion["k_pred_vol"] >= 0).all()
    check("Kronos 特征有限 & k_up_prob∈[0,1] & k_pred_vol≥0",
          bool(finite and up_ok and vol_ok))

    # (4) 标签正确性：独立用原始价重算 close.shift(-H)/close-1，与数据集逐行比对
    raw = px[["timestamps", "close"]].copy()
    raw["date"] = raw["timestamps"].dt.normalize()
    raw = raw.sort_values("date").reset_index(drop=True)
    raw["label_recmp"] = raw["close"].shift(-HORIZON) / raw["close"] - 1.0
    merged = fusion.merge(raw[["date", "label_recmp"]], on="date", how="left")
    max_diff = float((merged[label_col] - merged["label_recmp"]).abs().max())
    check("标签 = 未来H日真实收益（独立重算逐行一致）", max_diff < 1e-9,
          f"最大偏差={max_diff:.2e}")

    # (5) 无未来泄漏：原始价最后 H 行因无未来价应被丢弃，不出现在融合集
    last_h_dates = set(raw["date"].iloc[-HORIZON:])
    leaked = last_h_dates & set(fusion["date"])
    check("无未来泄漏（末尾H行无标签已剔除）", len(leaked) == 0,
          f"泄漏日期数={len(leaked)}")

    # (6) 1:1 对齐（日线日期唯一，无重复膨胀）
    check("特征日期唯一、融合无重复膨胀",
          fusion["date"].is_unique and len(fusion) <= len(kf),
          f"fusion={len(fusion)} kf={len(kf)}")

    # (7) 时间切分：不重叠 + 行数守恒
    no_overlap = (len(tr) == 0 or len(va) == 0 or len(te) == 0 or
                  (tr["date"].max() < va["date"].min() <= va["date"].max()
                   < te["date"].min()))
    conserved = (len(tr) + len(va) + len(te) == len(fusion))
    check("时间切分不重叠且行数守恒", bool(no_overlap and conserved),
          f"{len(tr)}+{len(va)}+{len(te)}={len(tr)+len(va)+len(te)} vs {len(fusion)}")

    # (8) 三个切分文件列结构一致
    cols_consistent = (list(tr.columns) == list(va.columns) == list(te.columns)
                       == list(fusion.columns))
    check("train/val/test 列结构完全一致", cols_consistent)

    print("-" * 72)
    print("样例（前 3 行）：")
    print(fusion[["date", "symbol"] + feat_cols + [label_col]].head(3).to_string(index=False))

    ok_all = all(checks)
    print("\n" + "=" * 72)
    print(f"校验结论：{'全部通过 ✅ 数据准确，可直接用于 C1 训练/验证/测试' if ok_all else '存在失败项 ❌ 见上'}")
    print("=" * 72)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
