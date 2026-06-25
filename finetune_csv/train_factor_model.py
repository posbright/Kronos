"""方案 B 训练入口：在 Kronos 主模型上微调「价量 + 因子条件旁路」。

中文说明：
    - 数据集 ``FactorKlineDataset``：在仓库 CSV 微调数据集基础上额外读取「因子列」，窗口切片时
      价格做 6 维 z-score 给 tokenizer，因子做单独 z-score（只用 lookback 段统计，防泄漏）。
    - 训练循环 ``train_factor``：tokenizer 仍只编码 6 维价格；factor 与 stamp 一起喂入
      ``KronosWithFactor``（见 factor_model.py）。与仓库 finetune_base_model.py 完全一致的
      teacher-forcing 损失（head.compute_loss）。

用法：
    # 真实使用（CSV 需含 timestamps + OHLCV(amount) + 你的因子列）
    python train_factor_model.py \
        --data-csv data/A_000001_with_factors.csv \
        --tokenizer pretrained/Kronos-Tokenizer-base \
        --predictor pretrained/Kronos-base \
        --factor-cols f_pe,f_pb,f_turnover,f_sent \
        --save-dir outputs/kronos_factor_000001 \
        --lookback 90 --pred 10 --epochs 3 --batch-size 16

    # 冒烟自测（无需任何预训练权重，用微型模型 + 合成数据跑通一个 epoch）
    python train_factor_model.py --smoke
"""

import argparse
import os
import sys

# 确保可以 `from model import ...` 与 `from factor_model import ...`
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import Dataset, DataLoader  # noqa: E402

from model import KronosTokenizer  # noqa: E402
from factor_model import KronosWithFactor, load_with_factor  # noqa: E402

_PRICE_COLS = ["open", "high", "low", "close", "volume", "amount"]
_TIME_COLS = ["minute", "hour", "weekday", "day", "month"]


class FactorKlineDataset(Dataset):
    """读取 CSV，窗口切片返回 (price, stamp, factor) 三元组（方案 B）。

    价格与因子各自做 z-score（仅用 lookback 段统计，防止未来信息泄漏）。
    """

    def __init__(self, data_path, factor_cols, data_type="train",
                 lookback_window=90, predict_window=10, clip=5.0,
                 train_ratio=0.7, val_ratio=0.15):
        self.factor_cols = list(factor_cols)
        self.data_type = data_type
        self.lookback_window = lookback_window
        self.predict_window = predict_window
        self.window = lookback_window + predict_window + 1
        self.clip = clip

        df = pd.read_csv(data_path)
        if "timestamps" not in df.columns:
            raise ValueError("CSV 必须包含 'timestamps' 列")
        df["timestamps"] = pd.to_datetime(df["timestamps"])
        df = df.sort_values("timestamps").reset_index(drop=True)

        missing = [c for c in _PRICE_COLS + self.factor_cols if c not in df.columns]
        if "amount" in missing:  # amount 缺失时补 0（与仓库 predict 行为一致）
            df["amount"] = 0.0
            missing.remove("amount")
        if missing:
            raise ValueError(f"CSV 缺少列: {missing}")

        df["minute"] = df["timestamps"].dt.minute
        df["hour"] = df["timestamps"].dt.hour
        df["weekday"] = df["timestamps"].dt.weekday
        df["day"] = df["timestamps"].dt.day
        df["month"] = df["timestamps"].dt.month

        cols = _PRICE_COLS + _TIME_COLS + self.factor_cols
        data = df[cols].copy()
        # 缺失值处理：先按时间前向填充（仅用过去信息，避免 bfill 引入未来泄漏），
        # 序列开头仍为空的用 0 兜底（z-score 后等价于“中性”取值）；最后断言无残留缺失。
        if data.isnull().any().any():
            data = data.ffill().fillna(0.0)
        assert not data.isnull().any().any(), "填充后仍存在缺失值，请检查因子列"

        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        if data_type == "train":
            data = data.iloc[:train_end]
        elif data_type == "val":
            data = data.iloc[train_end:val_end]
        else:
            data = data.iloc[val_end:]
        self.data = data.reset_index(drop=True)
        self.n_samples = max(0, len(self.data) - self.window + 1)
        print(f"[{data_type}] 数据长度 {len(self.data)}，可用样本 {self.n_samples}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        win = self.data.iloc[idx:idx + self.window]
        x = win[_PRICE_COLS].values.astype(np.float32)
        x_stamp = win[_TIME_COLS].values.astype(np.float32)
        factor = win[self.factor_cols].values.astype(np.float32)

        lb = self.lookback_window
        m, s = x[:lb].mean(0), x[:lb].std(0)
        x = np.clip((x - m) / (s + 1e-5), -self.clip, self.clip)

        fm, fs = factor[:lb].mean(0), factor[:lb].std(0)
        factor = np.clip((factor - fm) / (fs + 1e-5), -self.clip, self.clip)

        return (torch.from_numpy(x), torch.from_numpy(x_stamp), torch.from_numpy(factor))


def train_factor(model: KronosWithFactor, tokenizer: KronosTokenizer,
                 train_loader: DataLoader, val_loader, device: str,
                 epochs: int, lr: float = 2e-4, verbose: bool = True):
    """微调含因子旁路的 Kronos 主模型（单卡）。tokenizer 冻结、仅做编码。"""
    model = model.to(device)
    tokenizer = tokenizer.to(device).eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def _run_batch(batch, train: bool):
        batch_x, batch_stamp, batch_factor = (t.to(device) for t in batch)
        with torch.no_grad():
            s0, s1 = tokenizer.encode(batch_x, half=True)
        token_in = [s0[:, :-1], s1[:, :-1]]
        token_out = [s0[:, 1:], s1[:, 1:]]
        logits = model(token_in[0], token_in[1],
                       stamp=batch_stamp[:, :-1, :],
                       factor=batch_factor[:, :-1, :])
        loss, _, _ = model.head.compute_loss(logits[0], logits[1],
                                             token_out[0], token_out[1])
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
        return loss.item()

    history = []
    for epoch in range(epochs):
        model.train()
        tr_losses = [_run_batch(b, True) for b in train_loader]
        val_losses = []
        if val_loader is not None and len(val_loader) > 0:
            model.eval()
            with torch.no_grad():
                val_losses = [_run_batch(b, False) for b in val_loader]
        tr = float(np.mean(tr_losses)) if tr_losses else float("nan")
        va = float(np.mean(val_losses)) if val_losses else float("nan")
        history.append({"epoch": epoch + 1, "train_loss": tr, "val_loss": va})
        if verbose:
            print(f"[epoch {epoch + 1}/{epochs}] train_loss={tr:.4f} val_loss={va:.4f}")
    return history


def _smoke_test() -> None:
    """无需真实权重的端到端冒烟测试：合成 CSV -> 数据集 -> 微型模型 -> 训一个 epoch。"""
    import tempfile

    rng = np.random.default_rng(0)
    n = 400
    base = 10 + np.cumsum(rng.standard_normal(n) * 0.1)
    df = pd.DataFrame({
        "timestamps": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": base, "high": base + 0.2, "low": base - 0.2, "close": base + 0.05,
        "volume": rng.integers(1e5, 1e6, n).astype(float), "amount": 0.0,
        "f_pe": rng.standard_normal(n), "f_sent": rng.standard_normal(n),
    })
    factor_cols = ["f_pe", "f_sent"]

    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, "synth_factor.csv")
        df.to_csv(csv, index=False)

        train_ds = FactorKlineDataset(csv, factor_cols, "train",
                                      lookback_window=30, predict_window=5)
        val_ds = FactorKlineDataset(csv, factor_cols, "val",
                                    lookback_window=30, predict_window=5)
        assert len(train_ds) > 0 and len(val_ds) > 0
        # 校验三元组形状：window = 30 + 5 + 1 = 36
        x, stamp, factor = train_ds[0]
        assert x.shape == (36, 6), f"price 形状异常 {tuple(x.shape)}"
        assert stamp.shape == (36, 5), f"stamp 形状异常 {tuple(stamp.shape)}"
        assert factor.shape == (36, 2), f"factor 形状异常 {tuple(factor.shape)}"

        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=4, drop_last=True)

        torch.manual_seed(0)
        tokenizer = KronosTokenizer(
            d_in=6, d_model=32, n_heads=4, ff_dim=64, n_enc_layers=2, n_dec_layers=2,
            ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
            s1_bits=4, s2_bits=4, beta=1.0, gamma0=1.0, gamma=1.0, zeta=1.0, group_size=4,
        )
        model = KronosWithFactor(
            s1_bits=4, s2_bits=4, n_layers=2, d_model=32, n_heads=4, ff_dim=64,
            ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
            token_dropout_p=0.0, learn_te=True,
        )
        model.init_factor(factor_dim=len(factor_cols))

        history = train_factor(model, tokenizer, train_loader, val_loader,
                               device="cpu", epochs=1, verbose=False)
        assert len(history) == 1
        assert np.isfinite(history[0]["train_loss"]), "train_loss 非有限值"
    print(f"[smoke] train_factor_model 通过：数据集三元组形状正确，"
          f"训练 1 epoch train_loss={history[0]['train_loss']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="方案 B：微调含因子旁路的 Kronos 主模型")
    parser.add_argument("--data-csv", help="含 timestamps + OHLCV + 因子列的 CSV")
    parser.add_argument("--tokenizer", help="预训练 tokenizer 目录")
    parser.add_argument("--predictor", help="预训练主模型目录")
    parser.add_argument("--factor-cols", help="因子列名，逗号分隔，如 f_pe,f_pb,f_sent")
    parser.add_argument("--save-dir", help="微调后模型保存目录")
    parser.add_argument("--lookback", type=int, default=90)
    parser.add_argument("--pred", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default=None, help="cpu / cuda:0，默认自动")
    parser.add_argument("--smoke", action="store_true", help="运行无需权重的冒烟自测")
    args = parser.parse_args()

    if args.smoke:
        _smoke_test()
        return

    required = [args.data_csv, args.tokenizer, args.predictor, args.factor_cols, args.save_dir]
    if any(v is None for v in required):
        parser.error("非 --smoke 模式下必须提供 "
                     "--data-csv --tokenizer --predictor --factor-cols --save-dir")

    factor_cols = [c.strip() for c in args.factor_cols.split(",") if c.strip()]
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    train_ds = FactorKlineDataset(args.data_csv, factor_cols, "train",
                                  lookback_window=args.lookback, predict_window=args.pred)
    val_ds = FactorKlineDataset(args.data_csv, factor_cols, "val",
                                lookback_window=args.lookback, predict_window=args.pred)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, drop_last=True)

    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer)
    model = load_with_factor(args.predictor, factor_dim=len(factor_cols))

    train_factor(model, tokenizer, train_loader, val_loader, device=device,
                 epochs=args.epochs, lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    model.save_pretrained(args.save_dir)
    print(f"[train_factor_model] 已保存微调模型到 {args.save_dir}")


if __name__ == "__main__":
    main()
