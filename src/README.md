# src/ — module map

All live Python. Run modules from the repo root with `src` on the path:
`PYTHONPATH=src python -m utils.<module>`. Anything with a shell driver or SLURM
job is mapped in [../docs/SCRIPTS.md](../docs/SCRIPTS.md); the modules below with
neither are in the [Hand-run CLIs](#hand-run-clis) table at the bottom.

## Simulation core

- [llama_activate.py](llama_activate.py) — main ABM entry point: builds the vLLM
  pipe, constructs the SDA/SDC/Random network, runs the round loop, saves
  checkpoints, calls visualizations. `generate_parser()` defines the CLI used by
  `scripts/simulation/*.sh`.
- [classes/agent.py](classes/agent.py) — `Agent`: persona + PHQ-9 state, prompt
  construction, LLM output parsing, edge bookkeeping, well-being update.
- [classes/network.py](classes/network.py) — `_Network` / `RandomNetwork` /
  `SocialDistanceAttachment` (incl. SDC stub matching and power-law fitting).
  Owns the round loop, batched vLLM generation, both PHQ-9 assessment paths
  (LLM questionnaire and MentalBERT+MLP), and the opt-in bias correction.

## The three big utils/ modules

- [utils/visualization.py](utils/visualization.py) (~2.6k lines) — the plotting
  hub, in six blocks: network drawing (`print_network*`, the PHQ-9 filmstrip);
  CDS/distortion plots; embedding-trajectory & entrainment plots; PHQ-9
  structure (assortativity, neighbor correlation, bias/error); prompt-optimizer
  figures; and an eval-comparison CLI (`run_eval_comparison`, `__main__`) that
  generates the PNAS Tables 1–2 sources. Note: its network-evolution plots are
  superseded by `network_evolution.py` (below).
- [utils/network_evolution.py](utils/network_evolution.py) — corrected
  network-evolution figures for saved runs (CDS evolution, critical-slowing-down
  heatmaps, phase grids); recomputes CDS from raw text because the stored
  distortion flags are all-False. Reached via `tools/plot_network_evolution.py`
  (`scripts/plotting/run_plot_evolution.sh`), not by the simulation.
- [utils/prompt_optimizer.py](utils/prompt_optimizer.py) (~3.5k lines) — the
  TextGrad student–teacher prompt optimizer for post generation and PHQ-9
  assessment, plus the MentalBERT+MLP regressor training/eval stack. Full
  deep-dive: [../docs/prompt_optimizer.md](../docs/prompt_optimizer.md).

Also at this level: [utils/metrics.py](utils/metrics.py) (shared analysis
primitives — SBERT/MentalBERT embedding caches, CDS n-gram detection, TF-IDF,
PCA/UMAP, degree-weighted PHQ-9) and
[utils/eval_bert_on_csv.py](utils/eval_bert_on_csv.py) (score a saved
`regressor.pt` on a static CSV, offline mirror of the in-simulation BERT path).

## utils/ subpackages

- **tools/** — shared infrastructure + standalone analysis CLIs: the persona-pool
  loader `load_personas.py`, the run-directory layout singleton `path_manager.py`,
  the tweet/post vocabulary switch `format_config.py` (`FC`, most-imported module
  in the repo), network (de)serialization `reading_in.py`, the CDS validator
  `validate_cds.py` (canonical category-aware detector), the bias table
  `phq9_bias.py`, the regressor test-set replayer `build_bert_testset.py`
  (assessment-side), and four standalone plot scripts (see Hand-run CLIs).
- **create_data/** — the whole dataset story, from persona building to generation:
  `build_persona_phq9_eval.py` (step 0 — the canonical shared (persona, PHQ-9)
  eval file), `build_test_personas.py` / `build_finetune_personas.py` (persona
  sets for BERT fine-tuning), `test_phq9_llms.py` (`TestLLMs` harness used by
  every pipeline), `generate_test_data.py` (optimizer-aligned instructions, the
  SA workhorse), `generate_synthetic_dataset.py` (old framework, Agent-driven),
  `generate_posts_grok.py` (xAI API, standalone), `generate_posts_opt_h.py`
  (human-in-the-loop prompt iteration), `loaders.py` (shared loaders for these
  CLIs). Driver: `scripts/data_generation/create_data_menu.sh`.
- **sensitivity/** — 3-stage SA pipeline: generate (via `create_data`) → embed
  (`sa_embed.py`) → analyze (`sa_analyze.py`, the hub; `sa_phq9.py` regressor-space
  variant; prompt axis via `sa_analyze --prompt-reps`). `sa_network.py` is separate: Sobol SA +
  calibration of SDA/SDC topology parameters (CPU). `plot_network_targets{,_sdc}.py`
  plot simulated configs vs calibration target bands.
- **analyses/lexical_entrainment/** — `global/plot_lexical_entrainment.py`
  (per-seed MentalBERT entrainment trajectories; the `global` package name is a
  Python keyword, so it is runnable only via `python -m`, never importable) and
  `local/cds_entrainment.py` (local CDS entrainment vs a size-matched random
  baseline).

## Hand-run CLIs

No `.sh`/`.job` wrapper; run from the repo root with the venv python. Each
docstring carries the full argument example.

| Module | Run | Output |
|---|---|---|
| `utils.tools.plot_confusion_depression` | `./.venv_vllm/bin/python src/utils/tools/plot_confusion_depression.py` | `data/test_post/method_comparison/confusion_depression_classes.{png,csv}` |
| `utils.tools.plot_sbert_cosine_conditioning` | `./.venv_vllm/bin/python src/utils/tools/plot_sbert_cosine_conditioning.py` (first encode ~15 min CPU, cached) | `data/test_post/method_comparison/sbert_cosine_conditioning_seed35.{png,csv}` |
| `utils.tools.plot_phq9_mobility` | `PYTHONPATH=src ./.venv_vllm/bin/python -m utils.tools.plot_phq9_mobility` | `data/networks_post/basis/plots/phq9_mobility_*.png`, `mobility_phq9.csv` |
| `utils.tools.velocity_table` | `PYTHONPATH=src ./.venv_vllm/bin/python src/utils/tools/velocity_table.py` | console table + `plots/velocity_table.tex` |
| `utils.sensitivity.plot_network_targets` | `PYTHONPATH=src ./.venv_vllm/bin/python -m utils.sensitivity.plot_network_targets` | `data/sensitivity/network_target_ranges.png` |
| `utils.sensitivity.plot_network_targets_sdc` | same, `…plot_network_targets_sdc` | `data/sensitivity/network_target_ranges_sdc.png` |
| `utils.analyses.lexical_entrainment.local.cds_entrainment` | `PYTHONPATH=src ./.venv_vllm/bin/python -m utils.analyses.lexical_entrainment.local.cds_entrainment` | `plots/lexical_entrainment/local/entrainment_*.{csv,png}` |
| `utils.create_data.build_persona_phq9_eval` | `PYTHONPATH=src ./.venv_vllm/bin/python -m utils.create_data.build_persona_phq9_eval --out data/personas_eval_1000_phq9.csv …` | `data/personas_eval_1000_phq9.csv` (built ONCE; no-op if it exists) |
