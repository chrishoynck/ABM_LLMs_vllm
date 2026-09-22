"""Where the MentalBERT+MLP PHQ-9 assessors live, and what each was trained on.

Layout — one directory per *arm*, an arm being one training corpus:

    data/assessors/bert/<arm>/
        meta.json                  provenance: generator, generation prompt, corpus
        models/<model>_seed<NN>/   regressor.pt, performance.json, bias tables
        eval/on_<corpus>/          seed<NN>.csv + aggregate.csv, one dir per test corpus
        plots/

The arm name is `<generator>_<generation prompt>`. Those are two independent
axes so they get two slots: the old `bert_regression_finetuned_gemma4_minimal`
hid the fact that `gemma4` is a MODEL while `_minimal` is a PROMPT. `teacher` is
the odd one out, the un-fine-tuned baseline trained on the 12k-block teacher
corpus that every other arm is fine-tuned from.

Eval dirs reuse the arm vocabulary, so the estimator x corpus matrix behind the
paper's multi-model table is a glob instead of three naming schemes:

    data/assessors/bert/*/eval/on_gemma4_optimized/aggregate.csv

is "every assessor, scored on Gemma's held-out posts", and
`qwen27_optimized/eval/on_gemma4_optimized` is the cross-generator transfer cell.
"""
import glob
import json
import os

ROOT = "data/assessors/bert"

PROMPTS = {
    "minimal":   "data/prompt_optimization_h/qwen27_baseline/iter_0/prompt.txt",
    "optimized": "data/prompt_optimization_h/qwen27_baseline/iter_10/prompt.txt",
    "teacher":   "data/prompts_post.json",
}

# One entry per arm. `finetuned_from` None marks the baseline every other arm starts from.
ARMS = {
    "teacher": {
        "generator": "Qwen/Qwen3.5-27B", "generator_label": "Qwen3.5-27B",
        "model_short": "Qwen3.5-27B", "prompt": "teacher",
        "train_posts": "data/test_post/Qwen_Qwen3.5-27B/", "finetuned_from": None,
        "note": "12k-block teacher corpus, historical prompts_post.json (March 2026).",
    },
    "qwen27_optimized": {
        "generator": "Qwen/Qwen3.5-27B", "generator_label": "Qwen3.5-27B",
        "model_short": "Qwen3.5-27B", "prompt": "optimized",
        "train_posts": "data/finetune/qwen/train_posts_qwen.csv", "finetuned_from": "teacher",
        "note": "Deployed in the simulation: seed 35 is llama_activate.py's default regressor.",
    },
    "gemma4_optimized": {
        "generator": "google/gemma-4-31B-it", "generator_label": "Gemma-4-31B-it",
        "model_short": "gemma-4-31B-it", "prompt": "optimized",
        "train_posts": "data/finetune/gemma4/train_posts_gemma4.csv",
        "finetuned_from": "teacher", "note": "Multi-model arm.",
    },
    "qwen27_minimal": {
        "generator": "Qwen/Qwen3.5-27B", "generator_label": "Qwen3.5-27B",
        "model_short": "Qwen3.5-27B", "prompt": "minimal",
        "train_posts": "data/finetune/qwen_minimal/train_posts_qwen_minimal.csv",
        "finetuned_from": "teacher", "note": "Minimal-prompt arm (slurm 26885535).",
    },
    "gemma4_minimal": {
        "generator": "google/gemma-4-31B-it", "generator_label": "Gemma-4-31B-it",
        "model_short": "gemma-4-31B-it", "prompt": "minimal",
        "train_posts": "data/finetune/gemma4_minimal/train_posts_gemma4_minimal.csv",
        "finetuned_from": "teacher", "note": "Minimal-prompt arm (slurm 26885535).",
    },
    "mistral_optimized": {
        "generator": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "generator_label": "Mistral-Small-3.2-24B",
        "model_short": "Mistral-Small-3.2-24B-Instruct-2506", "prompt": "optimized",
        "train_posts": "data/finetune/mistral/train_posts_mistral.csv",
        "finetuned_from": "teacher", "note": "NOT GENERATED — listed so the figure skips it cleanly.",
    },
}

# Held-out post CSV behind each `eval/on_<corpus>/` directory.
CORPORA = {
    "qwen27_optimized":  "data/finetune/qwen/test_posts_qwen.csv",
    "gemma4_optimized":  "data/finetune/gemma4/test_posts_gemma4.csv",
    "qwen27_minimal":    "data/finetune/qwen_minimal/test_posts_qwen_minimal.csv",
    "gemma4_minimal":    "data/finetune/gemma4_minimal/test_posts_gemma4_minimal.csv",
    "mistral_optimized": "data/finetune/mistral/test_posts_mistral.csv",
    "calibration":       "data/finetune/qwen/calibration_posts_qwen.csv",
}

# Pre-2026-09-21 layout, kept so old job scripts, flags and notes still resolve.
LEGACY_DIRS = {
    "teacher":           "data/test_post/bert_regression",
    "qwen27_optimized":  "data/test_post/bert_regression_finetuned",
    "gemma4_optimized":  "data/test_post/bert_regression_finetuned_gemma4",
    "qwen27_minimal":    "data/test_post/bert_regression_finetuned_qwen_minimal",
    "gemma4_minimal":    "data/test_post/bert_regression_finetuned_gemma4_minimal",
    "mistral_optimized": "data/test_post/bert_regression_finetuned_mistral",
}
LEGACY_TAGS = {
    "qwen": "qwen27_optimized", "gemma4": "gemma4_optimized",
    "mistral": "mistral_optimized", "qwen_minimal": "qwen27_minimal",
    "gemma4_minimal": "gemma4_minimal",
}


def resolve(name):
    """Accept either a current arm/corpus name or a legacy GEN_TAG."""
    return LEGACY_TAGS.get(name, name)


def arm_dir(arm):
    return os.path.join(ROOT, resolve(arm))


def models_dir(arm):
    """Base dir holding {model_short}_seed{NN}/ — what --regressor-dir wants."""
    return os.path.join(arm_dir(arm), "models")


def plots_dir(arm):
    return os.path.join(arm_dir(arm), "plots")


def eval_dir(arm, corpus):
    """Where `arm`'s regressors scored on `corpus` land."""
    return os.path.join(arm_dir(arm), "eval", f"on_{resolve(corpus)}")


def regressor(arm, seed):
    arm = resolve(arm)
    return os.path.join(models_dir(arm), f"{ARMS[arm]['model_short']}_seed{seed}", "regressor.pt")


def available_arms():
    """Arms with at least one trained regressor on disk, in ARMS order."""
    return [a for a in ARMS if glob.glob(os.path.join(models_dir(a), "*_seed[0-9][0-9]", "regressor.pt"))]


def read_meta(arm):
    path = os.path.join(arm_dir(arm), "meta.json")
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def write_meta(arm, **extra):
    """Drop meta.json into an arm dir so the directory describes itself.

    Called by run_finetune.sh after training, and re-runnable by hand for the
    arms that predate it. `extra` records the run (seeds, lr, epochs, job id).
    """
    arm = resolve(arm)
    spec = ARMS[arm]
    meta = {
        "arm": arm,
        "generator": spec["generator"],
        "generator_label": spec["generator_label"],
        "generation_prompt": spec["prompt"],
        "generation_prompt_file": PROMPTS[spec["prompt"]],
        "train_posts": spec["train_posts"],
        "finetuned_from": spec["finetuned_from"],
        "model_short": spec["model_short"],
        "note": spec["note"],
        "legacy_dir": LEGACY_DIRS.get(arm),
    }
    meta.update(extra)
    os.makedirs(arm_dir(arm), exist_ok=True)
    path = os.path.join(arm_dir(arm), "meta.json")
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")
    return path
