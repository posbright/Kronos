"""方案 C 对比验证：在同一份融合数据上量化比较 C1（特征融合）与 C2（信号集成）。

中文说明：
    - C1 特征融合：把 Kronos 衍生特征 ⊕ 因子特征喂给一个下游模型，直接预测标签。
    - C2 信号集成：Kronos 特征单独出一路预测、因子特征单独出一路预测，再用
      (a) 加权（在验证集上搜索 alpha）或 (b) stacking（验证集上训一个线性元模型）融合。
    脚本在 train 上训基模型、在 val 上调组合器、在 test 上评估，输出 RMSE / IC / RankIC /
    方向命中率，并给出「谁更好」的结论。

    下游模型优先用 LightGBM（若已安装），否则回退到 numpy 实现的 Ridge 回归——因此本脚本
    在没有 lightgbm / sklearn 的环境也能完整跑通（冒烟自测即用回退实现）。

用法：
    # 真实使用（fusion_*.csv 由 build_fusion_dataset.py 生成）
    python compare_fusion_strategies.py \
        --train data/fusion_train.csv --val data/fusion_val.csv --test data/fusion_test.csv \
        --kronos-cols k_pred_ret,k_up_prob,k_pred_vol \
        --factor-cols pe,pb,roe,north_hold,news_sent,news_count,event_flag \
        --label label_fwd_ret_5d

    # 冒烟自测（无需任何文件 / 第三方库，用合成数据验证对比流程跑通）
    python compare_fusion_strategies.py --smoke
"""

import argparse
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
    """训练下游模型并对多个评估集预测。LightGBM 优先，否则 Ridge 回退。"""
    if _HAS_LGB:
        model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                  num_leaves=31, subsample=0.8,
                                  colsample_bytree=0.8, verbosity=-1)
        model.fit(x_tr, y_tr)
        return [model.predict(x) for x in x_eval_list]
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

    返回 (加权法 test 预测, stacking 法 test 预测, 最优 alpha)。
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
    weighted_te = best_alpha * _zscore(pk_te) + (1 - best_alpha) * _zscore(pf_te)

    # (b) stacking：在验证集上用线性元模型组合两路预测
    meta_x_va = np.column_stack([pk_va, pf_va])
    meta_x_te = np.column_stack([pk_te, pf_te])
    (stack_te,) = _ridge_fit_predict(meta_x_va, y_va, [meta_x_te], alpha=1.0)

    return weighted_te, stack_te, float(best_alpha)


# ----------------------------- 主流程 -----------------------------

def compare(tr, va, te, kronos_cols, factor_cols, label, verbose=True):
    """在同一数据上跑 C1 与 C2，返回各方案 test 指标与结论。"""
    feat_cols = list(kronos_cols) + list(factor_cols)
    y_te = te[label].values

    _, c1_te = run_c1(tr, va, te, feat_cols, label)
    w_te, s_te, alpha = run_c2(tr, va, te, kronos_cols, factor_cols, label)

    results = {
        "C1_特征融合": _metrics(y_te, c1_te),
        "C2_加权(alpha=%.2f)" % alpha: _metrics(y_te, w_te),
        "C2_stacking": _metrics(y_te, s_te),
    }
    # 以 test IC 作为主排序指标（越大越好），RMSE 作次要参考
    winner = max(results, key=lambda k: results[k]["IC"])

    if verbose:
        backend = "LightGBM" if _HAS_LGB else "Ridge(numpy 回退)"
        print(f"下游模型后端: {backend}")
        print(f"{'方案':<22}{'RMSE':>10}{'IC':>10}{'RankIC':>10}{'Hit':>8}")
        for name, m in results.items():
            print(f"{name:<22}{m['RMSE']:>10.4f}{m['IC']:>10.4f}"
                  f"{m['RankIC']:>10.4f}{m['Hit']:>8.3f}")
        print(f"==> 按 test IC 最优: {winner}")
    return results, winner


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
    results, winner = compare(tr, va, te, kronos_cols, factor_cols,
                              "label_fwd_ret_5d", verbose=True)

    # 校验：三方案指标均为有限值，IC 合法区间
    for name, m in results.items():
        assert np.isfinite(m["RMSE"]) and np.isfinite(m["IC"]), f"{name} 指标非有限"
        assert -1.0 <= m["IC"] <= 1.0, f"{name} IC 越界"
    assert winner in results
    print(f"[smoke] compare_fusion_strategies 通过：三方案均产出有效指标，最优={winner}")


def main():
    parser = argparse.ArgumentParser(description="方案 C：C1 特征融合 vs C2 信号集成 对比验证")
    parser.add_argument("--train"); parser.add_argument("--val"); parser.add_argument("--test")
    parser.add_argument("--kronos-cols", default="k_pred_ret,k_up_prob,k_pred_vol")
    parser.add_argument("--factor-cols",
                        default="pe,pb,roe,north_hold,news_sent,news_count,event_flag")
    parser.add_argument("--label", default="label_fwd_ret_5d")
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
    compare(tr, va, te, kronos_cols, factor_cols, args.label, verbose=True)


if __name__ == "__main__":
    main()
