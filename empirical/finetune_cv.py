"""Fine-tune the MentalBERT+MLP assessor on the empirical users, k-fold cross-validated by user.

Mirrors how the synthetic fine-tuned arms were made (`run_finetune.sh`): start from the
`teacher` regressor of each seed, AdamW at lr 2e-5, Huber loss (delta 6), 30 epochs,
best epoch on a validation split, one seed driving both the split and the init. The
input is the 10-tweet blocks of `score.block_centroids`, as in the synthetic training
data, but every split is by user: the synthetic arms had a separate 300-block test set,
which ~700 real users cannot spare, so every user is predicted once by a model
fine-tuned on the other folds (stratified by PHQ-9 band; 10% of each training fold's
users are the validation split), trained on those users' blocks. A user's prediction
is the mean over their blocks. A from-scratch baseline (same MLP, random init, lr 1e-4,
the repo's training default) shows what the synthetic pre-training adds. Writes
`<out-root>/finetuned_teacher/` and `<out-root>/scratch/` with the same seed<NN>.csv
columns as score.py, for bias.py.
Run: empirical/run_empirical.job (step 5).
"""

import argparse
import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from score import block_centroids, neural_net_BERT, user_means  # noqa: F401  (neural_net_BERT: unpickling)

TEACHER = "data/assessors/bert/teacher/models/{model}_seed{seed}/regressor.pt"
BAND_EDGES = [4, 9, 14, 19]            # PHQ-9 band upper edges, as sa_analyze.PHQ9_BANDS


def fresh_regressor(input_size: int) -> nn.Module:
    """Untrained regressor with the architecture and init of `prompt_optimizer.neural_net_BERT`."""
    reg = neural_net_BERT()
    reg.model = nn.Sequential(nn.Dropout(0.2), nn.Linear(input_size, 64), nn.ReLU(),
                              nn.Dropout(0.2), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    for m in reg.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            nn.init.constant_(m.bias, 0)
    return reg


def train(model, X, y, Xv, yv, lr, epochs=30, batch_size=8, weight_decay=1e-4):
    """The loop of `prompt_optimizer.train_bert`: AdamW, Huber(6), grad clip 1, LR on plateau, best val MAE.

    Args:
        model: regressor to train (modified in place).
        X, y: training centroids and PHQ-9 scores (tensors).
        Xv, yv: validation centroids and scores (tensors).
        lr (float): learning rate.
        epochs, batch_size, weight_decay: training settings.

    Returns:
        The model state with the lowest validation MAE.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.6, patience=2)
    loss_fn = nn.HuberLoss(delta=6.0)
    best, best_mae = copy.deepcopy(model), float("inf")
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss_fn(model(X[idx]).squeeze(-1), y[idx]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_mae = (model(Xv).squeeze(-1) - yv).abs().mean().item()
        sched.step(val_mae)
        if val_mae < best_mae:
            best, best_mae = copy.deepcopy(model), val_mae
    return best.eval()


def stratified_folds(true: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Fold index per user, dealt round-robin within each PHQ-9 band after a shuffle."""
    band = np.digitize(true, BAND_EDGES, right=True)
    fold = np.empty(len(true), dtype=int)
    for b in np.unique(band):
        idx = rng.permutation(np.where(band == b)[0])
        fold[idx] = np.arange(len(idx)) % k
    return fold


def main() -> None:
    """Parse args and write held-out predictions for every seed, fine-tuned and from scratch."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--emb", required=True, help="mentalbert_tweets.npz from empirical/embed.py.")
    parser.add_argument("--out-root", required=True, help="Writes <out-root>/{finetuned_teacher,scratch}/.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[34, 35, 36, 37, 38])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--model", default="Qwen3.5-27B", help="Teacher regressor subfolder prefix.")
    args = parser.parse_args()

    aids, true, baids, X = block_centroids(args.emb)
    user = pd.Index(aids).get_indexer(baids)           # user position of every block
    y = torch.as_tensor(true[user], dtype=torch.float32)
    print(f"[finetune] {len(aids)} users, {len(X)} blocks, {args.folds}-fold CV by user, seeds {args.seeds}")

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        fold = stratified_folds(true, args.folds, rng)
        raw = {"finetuned_teacher": np.empty(len(X)), "scratch": np.empty(len(X))}
        for k in range(args.folds):
            train_users = rng.permutation(np.where(fold != k)[0])
            n_val = max(1, len(train_users) // 10)
            val, tr = np.isin(user, train_users[:n_val]), np.isin(user, train_users[n_val:])
            test = fold[user] == k
            for arm, lr in (("finetuned_teacher", 2e-5), ("scratch", 1e-4)):
                if arm == "finetuned_teacher":
                    start = torch.load(TEACHER.format(model=args.model, seed=seed),
                                       map_location="cpu", weights_only=False)
                else:
                    start = fresh_regressor(X.shape[1])
                model = train(start, X[tr], y[tr], X[val], y[val], lr=lr)
                with torch.no_grad():
                    raw[arm][test] = model(X[test]).squeeze(-1).numpy()
        for arm, r_blocks in raw.items():
            r, n_blocks = user_means(baids, r_blocks, aids)
            pred = np.clip(np.round(r).astype(int), 0, 27)
            err = pred - true
            out_dir = os.path.join(args.out_root, arm)
            os.makedirs(out_dir, exist_ok=True)
            pd.DataFrame({"agent_id": aids, "true_phq9": true, "pred_phq9": pred, "raw_pred": r,
                          "abs_error": np.abs(err), "signed_bias": err, "n_blocks": n_blocks}
                         ).to_csv(os.path.join(out_dir, f"seed{seed}.csv"), index=False)
            print(f"[finetune] {arm} seed {seed}: MAE={np.abs(err).mean():.2f}  bias={err.mean():+.2f}  "
                  f"r={np.corrcoef(r, true)[0, 1]:+.2f}")


if __name__ == "__main__":
    main()
