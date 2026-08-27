# CS thesis — chapter/figure → code map

Thesis: *Evaluating Transformer Pipelines for the Generation and Assessment of
Depression in Social Media Data* (joint UvA-VU MSc Computer Science), repo
`~/thesis/Computer_Science_Transformer/`. Code repo (this one):
`https://github.com/chrishoynck/ABM_LLMs_vllm.git`. Paths relative to the
code-repo root. Doc index: [../README.md](../README.md).

| Thesis part | Repo component | Generator | Data inputs |
|---|---|---|---|
| Methods §Prompt Optimization; App. A Alg. 1; App. B §Prompt Optimization | `src/utils/prompt_optimizer.py` — see `prompt_optimizer.md` | `sbatch jobs/run_prompt_optimizer.job` (posts) / `run_prompt_optimizer_phq9.job` / `run_teacher_eval_iter0.job` | `data/test_post/Qwen_Qwen3.5-27B/`; outputs `data/test_post/optimized_{tweets,phq9}/` |
| Human-in-the-loop prompt variant (App. B) | `src/utils/create_data/generate_posts_opt_h.py` | per-iteration hand runs | `data/prompt_optimization_h/`; baselines `data/prompts_post_minimal.json`, `data/prompts_optimal.json` |
| Methods §Data Generation Loop; App. A Alg. 2 | `src/utils/create_data/{generate_test_data,test_phq9_llms,generate_synthetic_dataset,loaders}.py` | `bash scripts/data_generation/create_data_menu.sh` (uncomment ONE block) | personas from `src/utils/create_data/build_*.py` + `src/utils/tools/load_personas.py`; outputs `data/test_post/`, `data/test/Qwen/` (embeddings cache) |
| Methods §MentalBERT+MLP; App. A Alg. 3; App. B §Feature Extraction / §Fine-Tuning | `prompt_optimizer.py` (`train_BERT_model`, `neural_net_BERT`), `src/utils/eval_bert_on_csv.py`, `src/utils/tools/build_bert_testset.py` | `sbatch jobs/run_bert_optimizer.job`; `bash scripts/assessment/run_finetune.sh` | `data/finetune/`; outputs `data/test_post/bert_regression{,_finetuned}/` |
| Results §PHQ-9 Assessment / §Robustness & Cross-Domain | `src/utils/visualization.py` (eval-comparison CLI) | `bash scripts/assessment/run_eval_comparison.sh`, `run_phq9_on_bert_testset.sh`, `run_minimal_shift.sh` | `data/test_post/method_comparison/`, `optimized_phq9/*/eval_on_*` |
| Results §Context/State + §Decoding Sensitivity | `src/utils/sensitivity/{sa_embed,sa_analyze,sa_phq9}.py` | `bash scripts/sensitivity/sa_run.sh`, `sa_decoding_run.sh`; `sbatch jobs/sa_phq9_minimal_run.job` | `data/sensitivity/{agent,neighbor,joint,phq9,decoding,phq9_minimal_prompt}/` |
| App. B §Prompt Robustness (`SA_test_prompt`) | `sa_analyze.plot_prompt_sensitivity_pair` | `experiment.ipynb` cells 7-9 | `data/test_post/optimized_phq9/Qwen3.5-27B_sensitivity/prompt_sensitivity_pair.png` |
| (unused) prompt-variant SA | `sa_analyze --prompt-reps` (`sa_prompt.py` binned 2026-08-27) | `bash scripts/sensitivity/sa_prompt_run.sh`; `sbatch jobs/sa_prompt_baseline_run.job` | `data/prompt_optimization_h/qwen27_baseline/prompt_sa*` — appears in NO manuscript |
| Results §Cognitive Distortions + App. C §CDS Lexicon | `src/utils/tools/validate_cds.py` → `plots/cds_validation.png` | `PYTHONPATH=src python -m utils.tools.validate_cds` | `data/distorted_language_ngrams.tsv`, `data/finetune/cds_by_phq9*.csv` |
| App. C §Classification vs. Embedding Space | `src/utils/tools/plot_confusion_depression.py` + `plot_sbert_cosine_conditioning.py` | hand-run — see `../src/README.md` ("Hand-run CLIs") | `data/test_post/bert_regression/test_blocks_seed35.csv` |
| Methods fig `power_law_phq9` (HELIUS); App. B train/val curves | `experiment.ipynb` (savefig cells) → `plots/phq9_distribution.png` etc. | run the notebook | `data/confidential/phq9*.{sav,csv}`; optimizer/regressor logs |
| Methods §Hardware and Environment | `requirements_vllm.txt`, `jobs/*.job` (SLURM, 2×A100/H100) | — | — |

Not used by this thesis: the network simulation (`src/classes/network.py` round
loop, `data/networks_post/`), lexical entrainment (`src/utils/analyses/`),
`network_evolution.py`, `sa_network.py`, and the mobility/velocity/bias tools.

The narrative pipeline walk-through is [ROADMAP.md](ROADMAP.md) (stages tagged [CS]).

## PHQ-9 prompt-robustness baseline (corrected 2026-08-27)
Panel (a) of `SA_test_prompt` previously used the TextGrad optimizer's STARTING
prompt (from `data/prompts_post.json`) as the row labelled "minimal". It now uses the
true minimal instruction (`data/prompts_post_minimal.json`, 93 chars) with matching
held-out scores (`eval_on_test_blocks_seed35{,_minimal}`), so baseline and optimized
prompts are measured on the same blocks: minimal MAE 5.176 vs optimized 3.71-4.33.
Both appendix claims still hold, but check the wording:
  - "top-2 more similar to the baseline than to each other": HOLDS
    (cos(top1,top2)=0.596 < 0.631 / 0.628 to the baseline).
  - "optimized **much** higher similarity to one another than to the baseline":
    direction HOLDS but the gap shrank from +0.133 to **+0.041** (0.736 vs 0.695) —
    "much" is no longer well supported; consider softening.
The regenerated figure must be re-copied into the thesis as
`Figures/results/SA_test_prompt`.
