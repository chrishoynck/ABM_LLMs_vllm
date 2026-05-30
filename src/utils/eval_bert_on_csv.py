"""Evaluate a trained MentalBERT+MLP PHQ-9 regressor on a tweets_with_phq9 CSV.

Loads the regressor saved by `prompt_optimizer.train_BERT_model` (a pickled
neural_net_BERT module) and a sentence encoder, encodes each agent's posts
into the (mean ∥ max ∥ std) centroid the regressor was trained on, runs the
MLP, and reports MAE + signed bias overall and per ground-truth PHQ-9 score.

Mirrors `network._phq9_questionnaire_bert` but operates on a static CSV so
it can be pointed at any model's generated dataset.

Usage:
    python -m utils.eval_bert_on_csv \\
        --regressor data/test_post/bert_regression/Qwen3.5-27B_seed42/regressor.pt \\
        --csv data/grok_posts/posts_eval_grok_aligned.csv \\
        --out data/grok_posts/posts_eval_grok_aligned_bert.csv
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import torch

try:
    from utils.prompt_optimizer import parse_tweets_with_phq9_csv, neural_net_BERT  # noqa: F401
    from utils.metrics import generate_sbert_model
except ImportError:
    from .prompt_optimizer import parse_tweets_with_phq9_csv, neural_net_BERT  # noqa: F401
    from .metrics import generate_sbert_model


def evaluate(regressor_path: str, csv_path: str, out_path: str,
             mentalbert: bool = True, device: str = None) -> dict:
    """Run the regressor on every block in `csv_path` and write per-sample + per-PHQ-9 results."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    print(f"[bert-eval] loading regressor from {regressor_path}")
    regressor = torch.load(regressor_path, map_location=device, weights_only=False)
    regressor = regressor.to(device).eval()

    print(f"[bert-eval] loading sentence encoder (mentalbert={mentalbert})")
    encoder = generate_sbert_model(mentalbert=mentalbert).to(device)

    print(f"[bert-eval] parsing {csv_path}")
    tweet_blocks, true_answers, _personas, agent_ids = parse_tweets_with_phq9_csv(csv_path)
    print(f"[bert-eval] {len(tweet_blocks)} blocks parsed")

    centroids = []
    for block in tweet_blocks:
        valid = [t for t in block if t and t.upper() not in {"NO_POST", "NO_TWEET"}]
        if not valid:
            centroids.append(None)
            continue
        emb = encoder.encode(valid, convert_to_tensor=True).to(device)
        mean_v = emb.mean(dim=0)
        max_v = emb.max(dim=0)[0]
        var_emb = emb.var(dim=0)
        if torch.isnan(var_emb).any():
            var_emb = torch.zeros_like(var_emb)
        std_v = torch.sqrt(var_emb + 1e-8)
        centroids.append(torch.cat([mean_v, max_v, std_v], dim=0))

    keep_idx = [i for i, c in enumerate(centroids) if c is not None]
    if not keep_idx:
        raise RuntimeError("No block had any usable posts; nothing to evaluate.")
    batch = torch.stack([centroids[i] for i in keep_idx]).to(device)
    with torch.no_grad():
        raw_preds = regressor(batch).squeeze(-1).cpu().numpy()

    # Clamp to the PHQ-9 range and round to ints (same policy as the live BERT path).
    preds = np.clip(np.round(raw_preds).astype(int), 0, 27)

    abs_errors = []
    per_phq9_abs = defaultdict(list)
    per_phq9_signed = defaultdict(list)
    rows = []
    for k, csv_i in enumerate(keep_idx):
        true_v = int(true_answers[csv_i])
        pred_v = int(preds[k])
        err = pred_v - true_v
        abs_errors.append(abs(err))
        per_phq9_abs[true_v].append(abs(err))
        per_phq9_signed[true_v].append(err)
        rows.append({
            "agent_id": agent_ids[csv_i],
            "true_phq9": true_v,
            "pred_phq9": pred_v,
            "raw_pred": float(raw_preds[k]),
            "abs_error": abs(err),
            "signed_bias": err,
        })

    mae = float(np.mean(abs_errors))
    bias = float(np.mean([r["signed_bias"] for r in rows]))
    per_phq9 = {
        int(k): {
            "avg_mae": float(np.mean(per_phq9_abs[k])),
            "avg_bias": float(np.mean(per_phq9_signed[k])),
            "n_samples": len(per_phq9_abs[k]),
        }
        for k in sorted(per_phq9_abs)
    }
    skipped = len(tweet_blocks) - len(keep_idx)
    print(f"[bert-eval] overall MAE={mae:.3f}  bias={bias:+.3f}  "
          f"(n={len(rows)}; skipped {skipped} empty blocks)")
    for k, v in per_phq9.items():
        print(f"  PHQ-9={k:2d}  n={v['n_samples']:3d}  "
              f"MAE={v['avg_mae']:.3f}  bias={v['avg_bias']:+.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[bert-eval] per-sample predictions → {out_path}")

    summary_path = out_path.replace(".csv", "_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["phq9", "n_samples", "avg_mae", "avg_bias"])
        writer.writeheader()
        for k, v in per_phq9.items():
            writer.writerow({"phq9": k, **v})
    print(f"[bert-eval] per-PHQ-9 summary → {summary_path}")

    return {"mae": mae, "bias": bias, "per_phq9": per_phq9, "n": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--regressor", required=True,
                        help="Path to a regressor.pt saved by train_BERT_model.")
    parser.add_argument("--csv", required=True, dest="csv_path",
                        help="tweets_with_phq9 CSV to evaluate.")
    parser.add_argument("--out", required=True, dest="out_path",
                        help="Per-sample predictions CSV (a *_summary.csv sibling is also written).")
    parser.add_argument("--no-mentalbert", action="store_true",
                        help="Use generic SBERT instead of MentalBERT (must match training).")
    parser.add_argument("--device", default=None, help="torch device (default: auto).")
    args = parser.parse_args()
    evaluate(args.regressor, args.csv_path, args.out_path,
             mentalbert=not args.no_mentalbert, device=args.device)


if __name__ == "__main__":
    main()
