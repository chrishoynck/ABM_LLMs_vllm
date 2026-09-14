# src/ — module map

All live Python. Run modules from the repo root with `src` on the path:
`PYTHONPATH=src python -m utils.<module>`. Anything with a shell driver or SLURM
job is mapped in [../docs/SCRIPTS.md](../docs/SCRIPTS.md); the modules with
neither are in the [Hand-run CLIs](#hand-run-clis) table at the bottom. The small
"how good is X" checks live outside `src/`, in [../checks/](../checks/README.md).

## Docstring style
Every module, class and function has one docstring, Google style:
1. First line: what it does, one sentence, on the same line as the opening quotes.
2. Then, only if it changes how you use it, one to three plain sentences: why it exists, a gotcha, what it must match.
3. `Args:` one line per argument as `name (type): what it is` (type optional); `Returns:` unless it returns None.
4. No run commands, change logs or rejected designs (git has them). Commands live in the tables below and in docs/SCRIPTS.md.

## Simulation core

- [llama_activate.py](llama_activate.py): main ABM entry point. Builds the vLLM
  pipe and the SDA/SDC/Random network, runs the round loop, saves checkpoints and
  calls the plots. `generate_parser()` defines the CLI used by `scripts/simulation/*.sh`.
- [classes/agent.py](classes/agent.py): `Agent`, one persona plus PHQ-9 state,
  prompt building, LLM output parsing and edge bookkeeping.
- [classes/network.py](classes/network.py): `_Network` / `RandomNetwork` /
  `SocialDistanceAttachment` (SDA, and SDC stub matching with power-law fitting).
  Owns the round loop, batched vLLM generation, both PHQ-9 assessors (LLM
  questionnaire and MentalBERT+MLP) and the opt-in bias correction.

## The three big utils/ modules

- [utils/visualization.py](utils/visualization.py) (~2.3k lines): the plotting
  hub. Network drawing (`print_network_phq9`, `print_subnetworks_phq9`), CDS and
  entrainment plots, PHQ-9 structure (assortativity, neighbour correlation),
  prompt-optimizer figures, and the CLI
  `python -m utils.visualization {eval-comparison|multimodel}` that writes the
  PNAS Tables 1-2 sources and the generator x estimator table. Its old
  network-evolution plots are superseded by `network_evolution.py`.
- [utils/network_evolution.py](utils/network_evolution.py): network-evolution
  figures for saved runs (CDS evolution, critical-slowing-down heatmaps, phase
  grids). Recomputes CDS from the raw text with `tools/cds.py`. Reached through
  `tools/plot_network_evolution.py` (`scripts/plotting/run_plot_evolution.sh`).
- [utils/prompt_optimizer.py](utils/prompt_optimizer.py) (~3.4k lines): the
  TextGrad student-teacher prompt optimizer for post generation and PHQ-9
  assessment, plus the MentalBERT+MLP regressor training and eval. Deep dive:
  [../docs/prompt_optimizer.md](../docs/prompt_optimizer.md).

Also at this level: [utils/metrics.py](utils/metrics.py) (shared analysis
primitives: SBERT/MentalBERT embedding caches, flat CDS n-gram detection, TF-IDF,
PCA/UMAP, degree-weighted PHQ-9) and
[utils/eval_bert_on_csv.py](utils/eval_bert_on_csv.py) (score a saved
`regressor.pt` on a static CSV; its own file because `prompt_optimizer` imports
it lazily to avoid an import cycle).

## utils/ subpackages

- **tools/**: shared infrastructure plus standalone analysis CLIs. The persona
  loaders `load_personas.py`, the run-folder layout `path_manager.py`, the
  tweet/post vocabulary switch `format_config.py` (`FC`, imported almost
  everywhere), network (de)serialisation `reading_in.py`, the category-aware CDS
  detector `cds.py` (used by `network_evolution` and
  `checks/check_cds_tracks_phq9.py`), the bias table `phq9_bias.py`, the
  regressor test-set replayer `build_bert_testset.py`, and the hand-run plot
  scripts `plot_assessment_diagnostics.py`, `plot_phq9_mobility.py`,
  `velocity_table.py` and `plot_network_evolution.py`.
- **create_data/**: `build_personas.py` (the persona CSV builders as subcommands
  `eval` / `finetune` / `test-extra`), `test_phq9_llms.py` (the `TestLLMs`
  generation harness every pipeline uses; not a test), `generate_test_data.py`
  (the generator behind the SA runs and the fine-tune data),
  `generate_posts_opt_h.py` (human-in-the-loop prompt iteration) and `loaders.py`
  (shared loaders, model aliases, decoding defaults). Driver:
  `scripts/data_generation/create_data_menu.sh`.
- **sensitivity/**: 3-stage SA pipeline: generate (via `create_data`), embed
  (`sa_embed.py`), analyze (`sa_analyze.py`, the hub; `sa_phq9.py` repeats the
  axes in regressor space; the prompt axis via `sa_analyze --prompt-reps`).
  `sa_network.py` is separate: Sobol SA and calibration of the SDA/SDC topology
  (CPU). `plot_network_targets{,_sdc}.py` plot simulated configs against the
  calibration target bands.
- **analyses/lexical_entrainment/**: `global/plot_lexical_entrainment.py`
  (per-seed MentalBERT entrainment trajectories; the `global` package name is a
  Python keyword, so it only runs via `python -m`, never import it) and
  `local/cds_entrainment.py` (local CDS entrainment against a size-matched
  random baseline).

Small files kept on purpose, not merged: `eval_bert_on_csv.py` (import cycle),
`sa_embed.py` (pipeline stage with its own job), `generate_posts_opt_h.py` (the
human-in-the-loop arm), `format_config.py`, `path_manager.py`, `phq9_bias.py`,
`build_bert_testset.py` (CPU-only replay that must not import vLLM),
`velocity_table.py` and the two `plot_network_targets*.py` (clearer as
standalone scripts).

## Hand-run CLIs

No `.sh`/`.job` wrapper; run from the repo root with the venv python.

| Module | Run | Output |
|---|---|---|
| `utils.tools.plot_assessment_diagnostics` | `PYTHONPATH=src ./.venv_vllm/bin/python -m utils.tools.plot_assessment_diagnostics all` (first encode ~15 min CPU, cached) | `data/test_post/method_comparison/{confusion_depression_classes,sbert_cosine_conditioning_seed35}.{png,csv}` |
| `utils.tools.plot_phq9_mobility` | `PYTHONPATH=src ./.venv_vllm/bin/python -m utils.tools.plot_phq9_mobility` | `data/networks_post/basis/plots/phq9_mobility_*.png`, `mobility_phq9.csv` |
| `utils.tools.velocity_table` | `PYTHONPATH=src ./.venv_vllm/bin/python src/utils/tools/velocity_table.py` | console table + `plots/velocity_table.tex` |
| `utils.sensitivity.plot_network_targets` | `PYTHONPATH=src ./.venv_vllm/bin/python -m utils.sensitivity.plot_network_targets` | `data/sensitivity/network_target_ranges.png` |
| `utils.sensitivity.plot_network_targets_sdc` | same, `…plot_network_targets_sdc` | `data/sensitivity/network_target_ranges_sdc.png` |
| `utils.sensitivity.sa_phq9` | `PYTHONPATH=src ./.venv_vllm/bin/python -m utils.sensitivity.sa_phq9` (needs the `sa_embed` embeddings) | `data/sensitivity/plots_phq9/` |
| `utils.analyses.lexical_entrainment.local.cds_entrainment` | `PYTHONPATH=src ./.venv_vllm/bin/python -m utils.analyses.lexical_entrainment.local.cds_entrainment` | `plots/lexical_entrainment/local/entrainment_*.{csv,png}` |
| `utils.create_data.build_personas` | `PYTHONPATH=src ./.venv_vllm/bin/python -m utils.create_data.build_personas eval --out data/personas_eval_1000_phq9.csv` (built ONCE; no-op if it exists). `finetune` and `test-extra` are called by `scripts/assessment/run_finetune.sh` | `data/personas_eval_1000_phq9.csv` |
| `utils.visualization multimodel` | `PYTHONPATH=src ./.venv_vllm/bin/python -m utils.visualization multimodel` (CPU; also the last step of `run_llm_assessor_on_heldout.sh`) | `data/test_post/method_comparison/multimodel/` |
