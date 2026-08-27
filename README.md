# ABM_based_LLMS

Generative agent-based model (GABM) of depression contagion on social networks,
plus the PHQ-9 assessment pipeline it depends on. LLM agents (Qwen3.5-27B via
vLLM) write social-media posts from a persona + PHQ-9 profile; depression is
re-assessed from post histories by either an optimized LLM prompt or a
MentalBERT+MLP regressor; prompts are optimized with TextGrad. The repo grew
across two MSc theses (CS = generative/assessment model performance + sensitivity
analyses; GABM/CLS = network simulation) and the paper
`LLM_agent_Depression__PNAS_Nexus`.

## Layout

| Path | What it is |
|---|---|
| `src/` | all live Python — module map in [src/README.md](src/README.md) |
| `scripts/<stage>/` + `jobs/` | shell drivers + SLURM jobs — full map in [docs/SCRIPTS.md](docs/SCRIPTS.md) |
| `data/` | inputs + run outputs (~16 GB, gitignored) — see `data/NOTES.md` |
| `plots/` | figure outputs (gitignored) |
| `logs/` | run logs; nothing reads them back (gitignored) |
| `bin/` | recoverable discard from the 2026-08 cleanup — restore with `mv bin/X X`, see `bin/NOTES.md` |
| `experiment.ipynb` | notebook entry point for several figures |
| `docs/` | all documentation (index below) |

## Documentation

- [docs/ROADMAP.md](docs/ROADMAP.md) — narrative code roadmap: the pipeline stages, tagged with which manuscript (CS thesis / GABM thesis / PNAS paper) uses each
- [docs/STRUCTURE.md](docs/STRUCTURE.md) — folder-by-folder "what is used where" (live / historical / frozen / binned) + known quirks
- [docs/SCRIPTS.md](docs/SCRIPTS.md) — every driver script and SLURM job (inputs, outputs, GPU/walltime)
- [docs/prompt_optimizer.md](docs/prompt_optimizer.md) — TextGrad prompt optimization: method, file outline, how to run
- [docs/PAPER_MAP_PNAS.md](docs/PAPER_MAP_PNAS.md) — PNAS paper figure/table provenance
- [docs/THESIS_MAP_CS.md](docs/THESIS_MAP_CS.md) — CS thesis chapter → code map
- [docs/THESIS_MAP_GABM.md](docs/THESIS_MAP_GABM.md) — Computational Science (GABM) thesis chapter → code map
- [src/README.md](src/README.md) — module map + the hand-run analysis CLIs
- per-folder `NOTES.md` files (under `data/`, `bin/`) — file-level details for the data trees

## Getting started

```bash
uv venv .venv_vllm
source .venv_vllm/bin/activate
uv pip install -r requirements_vllm.txt
```

Python modules run from the repo root with `src` on the path:

```bash
PYTHONPATH=src python -m utils.<module> [...]
```

GPU work goes through SLURM: `sbatch jobs/<name>.job` (see [docs/SCRIPTS.md](docs/SCRIPTS.md)).

## Provenance

- PNAS paper figures/tables → [docs/PAPER_MAP_PNAS.md](docs/PAPER_MAP_PNAS.md)
- CS thesis (`Computer_Science_Transformer/`) → [docs/THESIS_MAP_CS.md](docs/THESIS_MAP_CS.md)
- GABM thesis (`Computational_Science_GABM/`) → [docs/THESIS_MAP_GABM.md](docs/THESIS_MAP_GABM.md)

### Known caveat — assessment and generation results come from different datasets

Three generation prompts produced the synthetic post data:

| Dataset | Generation prompt | Where |
|---|---|---|
| base / "high-fidelity" (~1.2k blocks) | `data/prompts_post.json` — an early, informally tuned prompt predating the systematic optimization study. NOT the minimal prompt; do not call it "non-optimized". | `data/test_post/Qwen_Qwen3.5-27B/` ([NOTES](data/test_post/Qwen_Qwen3.5-27B/NOTES.md)) |
| minimal | `data/prompts_post_minimal.json` | `data/sensitivity/phq9_minimal_prompt/` |
| human-optimized | `data/prompts_optimal.json` | `data/finetune/`, `data/sensitivity/phq9/`, all simulation runs |

Order actually run: high-fidelity set first → TextGrad optimization of the PHQ-9
*assessment* prompt on that set → TextGrad + human optimization of the
*generation* prompt, which was never fed back into a regenerated high-fidelity
set (no compute/time budget once the focus moved to the GABM thesis). The clean
order would have been: optimize generation → regenerate high-fidelity →
optimize assessment.

Consequences per pipeline:
- **LLM assessment prompt** (`data/test_post/optimized_phq9/`) — optimized and
  evaluated on the base set only; on human-optimized data it is *evaluated*
  (shift table) but never re-optimized.
- **MentalBERT+MLP** — trained on the base set (`bert_regression/`), then
  fine-tuned on human-optimized data (`bert_regression_finetuned/`, via
  `scripts/assessment/run_finetune.sh`). The fine-tuned regressor is the one the
  simulation uses, so GABM-thesis results are internally consistent.
- Per-band MAE/bias figures come from the base set; the S-BERT adjacent-band and
  class-similarity figures come from minimal + human-optimized SA runs
  (`data/sensitivity/`). **Any claim linking assessment error to linguistic
  overlap crosses distributions** unless it uses the base-set version.

Contained, not fatal: the bias direction (over-estimate mild, under-estimate
severe) and the rising adjacent-band similarity reproduce on every distribution
and in both assessor families, and the mismatch makes the shift table
(`scripts/assessment/run_eval_comparison.sh`) a fair symmetric OOD test rather
than a self-test. A within-distribution version of the linguistics↔error link
already exists: the S-BERT class-similarity matrix computed on the base-set BERT
test split (CS thesis App. B, `fig:phq9_confusion_cosim`).

Manuscript-side TODOs (fix there, not here) — agreed with D. Roy 2026-08-26 to
stay a background limitation, provided each result set names its dataset:
- Both Methods sections currently imply the high-fidelity set came from the
  optimized prompts; name its actual generating prompt where it is introduced.
- State the dataset in the assessment figure/table captions (CS thesis Figs
  4.6–4.8 + `tab:phq9-estimators`; paper `tab:estimators`,
  `fig:prompt_comparison`) and in Figs 4.3b / 4.4.
- Limitations: one sentence that both assessors were calibrated on the base set
  and only MentalBERT+MLP was re-fit and re-evaluated on the human-optimized
  one.
- Use one dataset name throughout; the CS thesis currently mixes "high fidelity",
  "non-optimized" and "the synthetic dataset" for the same data.
