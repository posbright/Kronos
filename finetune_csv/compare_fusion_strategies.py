"""方案 C 对比验证：在同一份融合数据上量化比较 C1（特征融合）与 C2（信号集成）。

中文说明（遵循方案 C 工程最佳实践：以 C1 为主线，C2 作为对照与兜底）：
    - C1 特征融合：把 Kronos 衍生特征 ⊕ 因子特征喂给一个下游模型，直接预测标签。
    - C2 信号集成：Kronos 特征单独出一路预测、因子特征单独出一路预测，再用
      (a) 加权（在验证集上搜索 alpha）或 (b) stacking（验证集上训一个线性元模型）融合。
    流程：train 训基模型 → val 调组合器并『选型』→ test 仅做最终评估。
    选型只看『验证集』指标（杜绝在测试集上选模型造成的选择泄漏）；并以 C1 为生产主线，
    仅当某个 C2 方案在『验证集 IC』上比 C1 高出 switch_threshold 时，才作为兜底切换上位。

    下游模型优先用 LightGBM（若已安装），否则回退到 numpy 实现的 Ridge 回归——因此本脚本
    在没有 lightgbm / sklearn 的环境也能完整跑通（冒烟自测即用回退实现）。

用法：
    # 真实使用（fusion_*.csv 由 build_fusion_dataset.py 生成）
    python compare_fusion_strategies.py \
        --train data/fusion_train.csv --val data/fusion_val.csv --test data/fusion_test.csv \
        --kronos-cols k_pred_ret,k_up_prob,k_pred_vol \
        --factor-cols pe,pb,roe,north_hold,news_sent,news_count,event_flag \
        --label label_fwd_ret_5d --switch-threshold 0.005 --out-json data/fusion_selection.json

    # 冒烟自测（无需任何文件 / 第三方库，用合成数据验证对比流程跑通）
    python compare_fusion_strategies.py --smoke
"""

import argparse
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# LightGBM 可选：装了就用，没装回退 numpy Ridge
try:
    import lightgbm as lgb  # noqa: E402
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False


# ----------------------------- 下游基模型 -----------------------------

def _ridge_fit_predict(x_tr, y_tr, x_eval_list, alpha=1.0):
    """numpy 闭式 Ridge 回归（带标准化），返回对每个 eval 矩阵的预测。

    用作没有 LightGBM 时的回退下游模型，也用于 stacking 的线性元模型。
    """
    x_tr = np.asarray(x_tr, dtype=np.float64)
    y_tr = np.asarray(y_tr, dtype=np.float64)
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-8
    xs = (x_tr - mu) / sd
    n, d = xs.shape
    xb = np.hstack([xs, np.ones((n, 1))])           # 加偏置列
    reg = alpha * np.eye(d + 1)
    reg[-1, -1] = 0.0                                # 不正则化偏置
    w = np.linalg.solve(xb.T @ xb + reg, xb.T @ y_tr)
    outs = []
    for x in x_eval_list:
        x = np.asarray(x, dtype=np.float64)
        xe = np.hstack([(x - mu) / sd, np.ones((len(x), 1))])
        outs.append(xe @ w)
    return outs


def _fit_predict(x_tr, y_tr, x_eval_list):
    """训练下游模型并对多个评估集预测。LightGBM（原生 API）优先，否则 Ridge 回退。"""
    if _HAS_LGB:
        dtrain = lgb.Dataset(np.asarray(x_tr, dtype=np.float64),
                             label=np.asarray(y_tr, dtype=np.float64))
        params = {"objective": "regression", "learning_rate": 0.05,
                  "num_leaves": 31, "feature_fraction": 0.8,
                  "bagging_fraction": 0.8, "bagging_freq": 1, "verbosity": -1}
        booster = lgb.train(params, dtrain, num_boost_round=300)
        return [booster.predict(np.asarray(x, dtype=np.float64)) for x in x_eval_list]
    return _ridge_fit_predict(x_tr, y_tr, x_eval_list, alpha=1.0)


# ----------------------------- 评估指标 -----------------------------

def _rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def _ic(y, p):
    """Pearson IC（信息系数）。"""
    y, p = np.asarray(y), np.asarray(p)
    if y.std() < 1e-12 or p.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(y, p)[0, 1])


def _rank_ic(y, p):
    """Spearman RankIC（对排序的相关），numpy 实现避免依赖 scipy。"""
    y = pd.Series(np.asarray(y)).rank().values
    p = pd.Series(np.asarray(p)).rank().values
    return _ic(y, p)


def _hit_rate(y, p):
    """方向命中率：预测涨跌方向与真实一致的比例。"""
    return float((np.sign(p) == np.sign(y)).mean())


def _metrics(y, p):
    return {"RMSE": _rmse(y, p), "IC": _ic(y, p),
            "RankIC": _rank_ic(y, p), "Hit": _hit_rate(y, p)}


def _zscore(a):
    a = np.asarray(a, dtype=np.float64)
    return (a - a.mean()) / (a.std() + 1e-8)


# ----------------------------- C1 / C2 -----------------------------

def run_c1(tr, va, te, feat_cols, label):
    """C1 特征融合：Kronos 特征 ⊕ 因子特征 -> 单个下游模型。"""
    p_va, p_te = _fit_predict(tr[feat_cols].values, tr[label].values,
                              [va[feat_cols].values, te[feat_cols].values])
    return p_va, p_te


def run_c2(tr, va, te, kronos_cols, factor_cols, label):
    """C2 信号集成：两路独立基模型 + (加权/stacking) 组合器。

    返回 ((加权 val 预测, 加权 test 预测), (stacking val 预测, stacking test 预测), 最优 alpha)。
    所有组合器参数（alpha / 元模型）只在验证集上确定，禁止触碰 test。
    """
    # 两路基模型分别在各自特征上训练，产出 val/test 预测
    pk_va, pk_te = _fit_predict(tr[kronos_cols].values, tr[label].values,
                                [va[kronos_cols].values, te[kronos_cols].values])
    pf_va, pf_te = _fit_predict(tr[factor_cols].values, tr[label].values,
                                [va[factor_cols].values, te[factor_cols].values])

    # (a) 加权：在验证集上搜索 alpha 使 IC 最大（禁止在 test 上调参）
    y_va = va[label].values
    best_alpha, best_ic = 0.5, -np.inf
    for alpha in np.linspace(0.0, 1.0, 21):
        sig_va = alpha * _zscore(pk_va) + (1 - alpha) * _zscore(pf_va)
        ic = _ic(y_va, sig_va)
        if ic > best_ic:
            best_ic, best_alpha = ic, alpha
    weighted_va = best_alpha * _zscore(pk_va) + (1 - best_alpha) * _zscore(pf_va)
    weighted_te = best_alpha * _zscore(pk_te) + (1 - best_alpha) * _zscore(pf_te)

    # (b) stacking：在验证集上用线性元模型组合两路预测
    meta_x_va = np.column_stack([pk_va, pf_va])
    meta_x_te = np.column_stack([pk_te, pf_te])
    stack_va, stack_te = _ridge_fit_predict(meta_x_va, y_va,
                                            [meta_x_va, meta_x_te], alpha=1.0)

    return (weighted_va, weighted_te), (stack_va, stack_te), float(best_alpha)


# ----------------------------- 选型与主流程 -----------------------------

# 生产主线策略名（方案 C 工程最佳实践：以 C1 特征融合为主线）
_MAIN_STRATEGY = "C1_特征融合"


def select_production(val_metrics, switch_threshold):
    """按『验证集 IC』选生产策略：C1 为主线，挑战者需超阈值才兜底切换。

    Args:
        val_metrics: {策略名: {"IC": ...}}，必须含主线 ``_MAIN_STRATEGY``。
        switch_threshold: 挑战者相对 C1 的验证集 IC 增益阈值（如 0.005）。
    Returns:
        被选为生产策略的名称（默认回到 C1，体现「以 C1 为主线」）。
    """
    c1_ic = val_metrics[_MAIN_STRATEGY]["IC"]
    challengers = {k: m["IC"] for k, m in val_metrics.items() if k != _MAIN_STRATEGY}
    if not challengers:
        return _MAIN_STRATEGY
    best = max(challengers, key=challengers.get)
    return best if challengers[best] >= c1_ic + switch_threshold else _MAIN_STRATEGY


def _print_block(title, metrics):
    """打印一组方案的指标表。"""
    print(f"\n[{title}]")
    print(f"{'方案':<22}{'RMSE':>10}{'IC':>10}{'RankIC':>10}{'Hit':>8}")
    for name, m in metrics.items():
        print(f"{name:<22}{m['RMSE']:>10.4f}{m['IC']:>10.4f}"
              f"{m['RankIC']:>10.4f}{m['Hit']:>8.3f}")


def compare(tr, va, te, kronos_cols, factor_cols, label,
            switch_threshold=0.005, verbose=True):
    """在同一数据上跑 C1 与 C2，按『验证集』选型，返回选型结果与 val/test 指标。

    工程最佳实践（方案 C）：
      - 以 C1 特征融合为主线（默认生产策略）。
      - C2（加权 / stacking）作为对照与兜底；仅当其『验证集 IC』≥ C1 + switch_threshold
        时才切换上位，避免噪声波动导致频繁换线。
      - 选型只看验证集，test 仅做最终评估，杜绝选择泄漏。
    """
    feat_cols = list(kronos_cols) + list(factor_cols)
    y_va, y_te = va[label].values, te[label].values

    c1_va, c1_te = run_c1(tr, va, te, feat_cols, label)
    (w_va, w_te), (s_va, s_te), alpha = run_c2(
        tr, va, te, kronos_cols, factor_cols, label)

    preds = {
        _MAIN_STRATEGY: (c1_va, c1_te),
        "C2_加权(alpha=%.2f)" % alpha: (w_va, w_te),
        "C2_stacking": (s_va, s_te),
    }
    val_metrics = {k: _metrics(y_va, p[0]) for k, p in preds.items()}
    test_metrics = {k: _metrics(y_te, p[1]) for k, p in preds.items()}

    # 以验证集 IC 选生产策略：C1 主线 + 兜底切换
    production = select_production(val_metrics, switch_threshold)

    if verbose:
        backend = "LightGBM" if _HAS_LGB else "Ridge(numpy 回退)"
        print(f"下游模型后端: {backend}    选型阈值(val IC 增益): +{switch_threshold:.3f}")
        _print_block("验证集 (val) —— 仅用于选型", val_metrics)
        _print_block("测试集 (test) —— 仅最终评估", test_metrics)
        c1_ic = val_metrics[_MAIN_STRATEGY]["IC"]
        prod_ic = val_metrics[production]["IC"]
        if production == _MAIN_STRATEGY:
            print(f"\n==> 生产策略: {production}（主线；无 C2 方案在 val 上超过 "
                  f"C1 IC {c1_ic:.4f} + {switch_threshold:.3f}）")
        else:
            print(f"\n==> 生产策略: {production}（兜底切换；val IC {prod_ic:.4f} "
                  f"≥ C1 {c1_ic:.4f} + {switch_threshold:.3f}）")
        print(f"    生产策略测试集指标: {test_metrics[production]}")

    return {
        "production": production,
        "alpha": alpha,
        "switch_threshold": switch_threshold,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def _synth_fusion(n=900, seed=0):
    """生成含『因子×价量交互』的合成融合数据，用于冒烟自测。

    标签同时依赖 Kronos 特征、因子特征及其交互项 -> C1 理应有优势。
    """
    rng = np.random.default_rng(seed)
    k_pred_ret = rng.standard_normal(n) * 0.01
    k_up_prob = 1 / (1 + np.exp(-(k_pred_ret * 60 + rng.standard_normal(n) * 0.5)))
    k_pred_vol = np.abs(rng.standard_normal(n)) * 0.01
    pe = 15 + rng.standard_normal(n)
    pb = 1.5 + rng.standard_normal(n) * 0.2
    roe = 0.1 + rng.standard_normal(n) * 0.02
    news_sent = rng.standard_normal(n)
    news_count = rng.integers(0, 10, n).astype(float)
    event_flag = (rng.random(n) > 0.85).astype(float)

    label = (1.2 * k_pred_ret
             + 0.004 * (k_up_prob - 0.5)
             + 0.05 * _zscore(roe)            # 因子主效应
             - 0.03 * _zscore(pe)
             + 0.06 * k_pred_ret * _zscore(roe)   # 交互项：C1 才吃得到
             + 0.01 * news_sent
             + rng.standard_normal(n) * 0.012)    # 噪声

    df = pd.DataFrame({
        "k_pred_ret": k_pred_ret, "k_up_prob": k_up_prob, "k_pred_vol": k_pred_vol,
        "pe": pe, "pb": pb, "roe": roe, "news_sent": news_sent,
        "news_count": news_count, "event_flag": event_flag,
        "label_fwd_ret_5d": label,
    })
    # 按行序（模拟时间序）切 60/20/20
    a, b = int(n * 0.6), int(n * 0.8)
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def _smoke_test():
    """无需文件 / 第三方库的端到端冒烟测试。"""
    tr, va, te = _synth_fusion(n=900)
    kronos_cols = ["k_pred_ret", "k_up_prob", "k_pred_vol"]
    factor_cols = ["pe", "pb", "roe", "news_sent", "news_count", "event_flag"]
    out = compare(tr, va, te, kronos_cols, factor_cols,
                  "label_fwd_ret_5d", switch_threshold=0.005, verbose=True)

    # 校验：val/test 指标均为有限值，IC 合法区间
    for split in ("val_metrics", "test_metrics"):
        for name, m in out[split].items():
            assert np.isfinite(m["RMSE"]) and np.isfinite(m["IC"]), f"{name} 指标非有限"
            assert -1.0 <= m["IC"] <= 1.0, f"{name} IC 越界"
    assert out["production"] in out["val_metrics"]
    assert _MAIN_STRATEGY in out["val_metrics"], "缺少主线 C1"

    # 显式校验「主线 + 阈值切换」逻辑：挑战者需超阈值才能上位
    vm = {_MAIN_STRATEGY: {"IC": 0.100},
          "C2_加权(alpha=0.50)": {"IC": 0.102},
          "C2_stacking": {"IC": 0.104}}
    assert select_production(vm, 0.005) == _MAIN_STRATEGY, "未达阈值不应切换（应留在 C1）"
    vm2 = dict(vm, **{"C2_stacking": {"IC": 0.110}})
    assert select_production(vm2, 0.005) == "C2_stacking", "超过阈值应兜底切换到挑战者"

    print(f"[smoke] compare_fusion_strategies 通过：选型在验证集完成（无 test 泄漏），"
          f"主线=C1，生产策略={out['production']}")


def main():
    parser = argparse.ArgumentParser(description="方案 C：C1 特征融合 vs C2 信号集成 对比验证")
    parser.add_argument("--train"); parser.add_argument("--val"); parser.add_argument("--test")
    parser.add_argument("--kronos-cols", default="k_pred_ret,k_up_prob,k_pred_vol")
    parser.add_argument("--factor-cols",
                        default="pe,pb,roe,north_hold,news_sent,news_count,event_flag")
    parser.add_argument("--label", default="label_fwd_ret_5d")
    parser.add_argument("--switch-threshold", type=float, default=0.005,
                        help="C2 相对 C1 的验证集 IC 增益阈值，超过才切换上位（默认 0.005）")
    parser.add_argument("--out-json", default=None, help="可选：将选型结果与指标落盘为 JSON")
    parser.add_argument("--smoke", action="store_true", help="运行无需文件的冒烟自测")
    args = parser.parse_args()

    if args.smoke:
        _smoke_test()
        return

    if not all([args.train, args.val, args.test]):
        parser.error("非 --smoke 模式下必须提供 --train --val --test")

    tr = pd.read_csv(args.train)
    va = pd.read_csv(args.val)
    te = pd.read_csv(args.test)
    kronos_cols = [c.strip() for c in args.kronos_cols.split(",") if c.strip()]
    factor_cols = [c.strip() for c in args.factor_cols.split(",") if c.strip()]
    # 仅保留实际存在的因子列，避免某些标的缺列报错
    factor_cols = [c for c in factor_cols if c in tr.columns]
    out = compare(tr, va, te, kronos_cols, factor_cols, args.label,
                  switch_threshold=args.switch_threshold, verbose=True)

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[compare] 选型结果已写入 {args.out_json}")


if __name__ == "__main__":
    main()
