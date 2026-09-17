# ABM_based_LLMS

Code for a line of research on **depression contagion in social networks**: does
depressive language spread between connected people, and what does that do to the
mental-health distribution of a population? The answer is pursued with a generative
agent-based model (GABM) in which LLM agents write social-media posts from a persona
and a PHQ-9 depression profile, read their neighbours' posts, and then have their own
severity re-assessed *from the language they produce* — so depression evolves through
text rather than through a hand-written update rule.

That simulation only means something if two components underneath it hold up, and both
are built and validated here as well: a **generator** whose synthetic posts genuinely
reflect a target PHQ-9 score, and an **assessor** that can recover a PHQ-9 score back
out of a block of posts. Building and stress-testing those two is the CS thesis; running
the network simulation on top of them is the Computational Science (GABM) thesis; the
paper `LLM_agent_Depression__PNAS_Nexus` draws on both.

Stack: Qwen3.5-27B served with vLLM for generation and LLM-based assessment, a supervised
MentalBERT+MLP regressor as the second assessor, prompts optimized with TextGrad plus a
human-in-the-loop arm. Everything GPU-bound runs through SLURM.

## Main pipelines

1. **Synthetic data generation** — persona (PersonaHub) × PHQ-9 target drawn from the
   heavy-tailed HELIUS distribution (power-law fit γ=2.36) → labelled post blocks.
2. **Prompt optimization** — TextGrad student–teacher loops over both the post-generation
   and the PHQ-9-assessment prompt, plus a human-in-the-loop arm that beats TextGrad on
   the shared teacher-scored test set.
3. **PHQ-9 assessment** — two architecturally unrelated assessors compared on identical
   held-out data: LLM prompt inference (minimal vs optimized) and the MentalBERT+MLP
   regressor, incl. a base → human-optimized distribution-shift test.
4. **Sensitivity analyses** — stochastic axes (agent persona / neighbour context / joint)
   against an irreducible-noise baseline, PHQ-9 severity-band separability in embedding
   space (S-BERT adjacent-band cosine), and a temperature/top-p decoding grid.
5. **Network construction & calibration** — SDA network over latent dims + age + PHQ-9 and
   its scale-free SDC extension, Sobol-calibrated against empirical target bands.
6. **Simulation** — round loop of activation → post generation from persona + own and
   neighbour history → PHQ-9 re-assessment every 10 rounds.
7. **Simulation analysis** — trajectories and phase plots, lexical entrainment (global
   MentalBERT drift, local CDS entrainment vs a random baseline), per-agent PHQ-9 mobility,
   velocity of contagion.
8. **Empirical validation** — generated language against the Bathina et al. cognitive-
   distortion lexicon (r = +0.95 with severity) and embedding-overlap diagnostics.

Stage-by-stage file map, tagged with the manuscript that uses each:
[docs/ROADMAP.md](docs/ROADMAP.md).

## Layout

| Path | What it is |
|---|---|
| `src/` | all live Python — module map in [src/README.md](src/README.md) |
| `scripts/<stage>/` + `jobs/` | shell drivers + SLURM jobs — full map in [docs/SCRIPTS.md](docs/SCRIPTS.md) |
| `checks/` | small "how good is X" checks with their own jobs (surface cues, multi-model smoke run, CDS vs PHQ-9), see [checks/README.md](checks/README.md) |
| `data/` | inputs + run outputs — non-confidential research data is tracked in git; `confidential/` (HELIUS), `methodology_paper/`, `networks_post/` (14 GB) and `test/` stay local. **Provenance map (what comes from where, used for what): [data/README.md](data/README.md)**; cleanup/status notes in `data/NOTES.md` |
| `plots/` | figure outputs (gitignored) |
| `logs/` | run logs; nothing reads them back (gitignored) |
| `bin/` | recoverable discard from the 2026-08 and 2026-09 cleanups — restore with `mv bin/X X`, see `bin/NOTES.md` |
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
- [src/README.md](src/README.md) — module map, the docstring style, the hand-run analysis CLIs
- [checks/README.md](checks/README.md) — the small quality checks: what each one answers and how to run it
- [data/README.md](data/README.md) — data provenance map: where every dataset comes from (real vs hand-made vs generated, and by which script) and what it is used for
- per-folder `NOTES.md` files (under `data/`, `bin/`) — file-level status/cleanup details for the data trees

## Getting started

```bash
uv venv .venv_vllm
source .venv_vllm/bin/activate
uv pip install -r requirements_vllm.txt
```

The non-Qwen student generators (Gemma-4-31B-it, Mistral-Small-3.2-24B) run in
a second venv with a newer vLLM and transformers, used only for post generation:

```bash
uv venv --python 3.10 .venv_vllm_g4
uv pip install --python .venv_vllm_g4/bin/python -r requirements_vllm_g4.txt
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

## Known caveat — assessment and generation results come from different datasets

Three generation prompts produced the synthetic post data: the **base** ("high-fidelity")
set of 12k blocks / 120k posts from an early, informally tuned prompt
(`data/prompts_post.json`; not the minimal prompt, so don't call it "non-optimized"), the
**minimal** baseline (`data/prompts_post_minimal.json`), and the **human-optimized** set
(`data/prompts_optimal.json`) used for the linguistic analyses and every simulation run.
The base set came first, so both assessors were calibrated on it; only MentalBERT+MLP was
later re-fit on human-optimized data. Consequence: per-band assessment-error figures come
from the base set while the S-BERT/linguistic figures come from minimal + human-optimized
runs, so **any claim linking assessment error to linguistic overlap crosses distributions**
unless it uses the base-set version (which exists: CS thesis App. B, `fig:phq9_confusion_cosim`).

**The simulation is unaffected.** The fine-tuned regressor
(`data/test_post/bert_regression_finetuned/`, via `scripts/assessment/run_finetune.sh`) was
fine-tuned on human-optimized data — the same distribution the optimized generative
pipeline produces at simulation time — so the assessor the GABM uses is *in distribution*
with the posts it scores, and the GABM-thesis results are internally consistent. Elsewhere
the mismatch is contained: the bias direction (over-estimate mild, under-estimate severe)
and the rising adjacent-band similarity reproduce on every distribution and in both
assessor families, and the mismatch makes the shift table
(`scripts/assessment/run_eval_comparison.sh`) a fair symmetric OOD test rather than a
self-test.

**Manuscript-side TODOs** (fix in the manuscripts, not here) — agreed with D. Roy
2026-08-26 to stay a background limitation, provided each result set names its dataset:
name the base set's actual generating prompt where it is introduced; state the dataset in
the assessment figure/table captions (CS thesis Figs 4.3b/4.4/4.6–4.8 + `tab:phq9-estimators`;
paper `tab:estimators`, `fig:prompt_comparison`); add one limitations sentence that both
assessors were calibrated on the base set and only MentalBERT+MLP was re-fit and
re-evaluated on the human-optimized one; use a single dataset name throughout (the CS
thesis mixes "high fidelity", "non-optimized" and "the synthetic dataset"); and fix the CS
thesis Methods 10× slip — "≈1,200 post-blocks (≈12,000 posts)" should be 12,000 blocks /
120,000 posts (splits 9,638+1,237+1,125), as the PNAS manuscript already has it.
