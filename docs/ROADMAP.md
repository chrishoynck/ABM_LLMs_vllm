# Code roadmap

The pipeline stages of the project, tagged with the manuscripts that use them:
**[CS]** = CS thesis (`Computer_Science_Transformer/`), **[GABM]** = Computational
Science thesis (`Computational_Science_GABM/`), **[PNAS]** = the paper
(`LLM_agent_Depression__PNAS_Nexus/`). Paths relative to the repo root. Exact
chapter/figure → code tables: [THESIS_MAP_CS.md](THESIS_MAP_CS.md),
[THESIS_MAP_GABM.md](THESIS_MAP_GABM.md), [PAPER_MAP_PNAS.md](PAPER_MAP_PNAS.md).

## 1. Synthetic data generation — [CS] [PNAS]

`src/utils/create_data/`: `build_persona_phq9_eval.py` builds the shared
(persona, PHQ-9) eval file, `test_phq9_llms.py` (`TestLLMs`) is the network-free
generation harness, `generate_test_data.py` produces the labelled post blocks
(~1,200 blocks / ~12,000 posts, `data/test_post/`). Personas from PersonaHub via
`tools/load_personas.py`; PHQ-9 targets grounded in the heavy-tailed HELIUS
distribution (power-law fit γ=2.36, `experiment.ipynb` → `plots/phq9_distribution.png`).
Driver: `scripts/data_generation/create_data_menu.sh`.

## 2. Prompt optimization — [CS] [PNAS]

`src/utils/prompt_optimizer.py` optimizes both the post-generation and the
PHQ-9-assessment prompt with TextGrad in a student–teacher loop
(`jobs/run_prompt_optimizer{,_phq9}.job`); method + how-to in
[prompt_optimizer.md](prompt_optimizer.md). The human-in-the-loop variant
(`create_data/generate_posts_opt_h.py` + `data/prompt_optimization_h/`) is the
paper's second optimizer arm — it beats TextGrad on the shared teacher-scored
test set.

## 3. PHQ-9 assessment pipelines — [CS] [PNAS] [GABM]

Two architecturally unrelated assessors, compared on identical held-out data:
(a) LLM prompt inference (minimal vs TextGrad-optimized, stage 2); (b) the
supervised MentalBERT+MLP regressor — `train_BERT_model` in
`prompt_optimizer.py` (`jobs/run_bert_optimizer.job`), fine-tuning
`scripts/assessment/run_finetune.sh` [CS], offline scoring
`src/utils/eval_bert_on_csv.py`. Shared test set via `tools/build_bert_testset.py`
+ `scripts/assessment/run_phq9_on_bert_testset.sh`; distribution shift (base →
human-optimized dataset) via `run_minimal_shift.sh`; the estimator-comparison
figures/tables (PNAS Tables 1–2) via `run_eval_comparison.sh` →
`utils.visualization`. The regressor also runs *inside* the GABM simulation,
with the per-level bias correction from `tools/phq9_bias.py`
(`jobs/run_bias_calibration.job`) [GABM].

## 4. Sensitivity analyses — [CS] [PNAS]

`src/utils/sensitivity/`: encode runs with `sa_embed.py`, then `sa_analyze.py`
for the stochastic axes (agent persona / neighbour context / joint vs the
irreducible-noise baseline), the PHQ-9 severity-band separability (S-BERT
adjacent-band cosine — the paper's convergence result), and the
temperature/top-p decoding grid (PNAS `decoder_SA`); `sa_phq9.py` repeats the
axes in regressor space (PHQ-9-drop anchors figure); `sa_prompt.py` prompt
robustness [CS only]. Drivers in `scripts/sensitivity/`.

## 5. Network construction & calibration — [GABM]

`src/classes/network.py` builds the SDA network (social-distance attachment over
latent dims + age + PHQ-9) and its scale-free SDC extension (power-law stub
matching). Calibration: `sensitivity/sa_network.py` (Saltelli/Sobol,
`scripts/sensitivity/run_sa_network.sh`), checked against empirical target bands
by `sensitivity/plot_network_targets{,_sdc}.py`.

## 6. Simulation — [GABM]

`src/llama_activate.py` is the entry point; the round loop lives in
`classes/network.py` + `classes/agent.py` (activation → post generation from
persona + own/neighbour history → PHQ-9 re-assessment every 10 rounds, stage 3).
Drivers: `scripts/simulation/run_simulation_{sda,sdc}.sh` +
`jobs/run_simulation_{sda,sdc}.job`; runs land in `data/networks_post/`
(`net.json` checkpoints). Happy-hub variants configure there too
(`data/happy_persona.csv`).

## 7. Simulation analysis & metrics — [GABM]

Per saved run: trajectories/phase plots (`utils/network_evolution.py` via
`tools/plot_network_evolution.py`, `scripts/plotting/run_plot_evolution.sh`, and
the phase-space cells in `experiment.ipynb`); lexical entrainment — global
MentalBERT drift (`analyses/lexical_entrainment/global/`,
`jobs/run_lexical_entrainment.job`) and local CDS entrainment vs a random
baseline (`local/cds_entrainment.py`); mobility (per-agent PHQ-9 RMSSD,
`tools/plot_phq9_mobility.py`); velocity of contagion (`tools/velocity_table.py`
→ `plots/velocity_table.tex`).

## 8. Empirical validation & figures — [CS] [PNAS]

`tools/validate_cds.py` tests generated language against the Bathina et al.
cognitive-distortion n-gram lexicon, per category (the paper's r=+0.95 severity
correlation and the per-category divergences) → `plots/cds_validation.png`.
Appendix-level diagnostics: `tools/plot_confusion_depression.py` +
`tools/plot_sbert_cosine_conditioning.py` (assessment error vs embedding-space
overlap). `experiment.ipynb` holds the HELIUS power-law figure and the
train/val-curve cells.
