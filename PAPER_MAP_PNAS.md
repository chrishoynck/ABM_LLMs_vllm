# LLM_agent_Depression__PNAS_Nexus — figure/table provenance

Repo-generated assets in the paper (`manuscript.tex`), and how to regenerate them.

| Paper asset | Repo artifact | Generator | Data inputs |
|---|---|---|---|
| Fig `decoder_SA.jpg` (§2.2) | `data/sensitivity/plots_sbert/decoding_phq9_linearity.png` | `PYTHONPATH=src python -m utils.sensitivity.sa_analyze --emb-name embeddings_sbert.npz` | `data/sensitivity/decoding/setting_*/rep_*/` |
| Fig `SA-PHQ9.png` (§2.2) | `data/sensitivity/plots_sbert/agent_phq9_combined.png` | same `sa_analyze` run | `data/sensitivity/phq9/*` + `phq9_minimal_prompt/*` |
| Fig `syntetic_data_CDS.png` (§2.3) | `plots/cds_validation.png` | `PYTHONPATH=src python -m utils.tools.validate_cds` | `data/finetune/cds_by_phq9*.csv`, `data/distorted_language_ngrams.tsv` |
| Fig `power_law_phq9.jpg` (§4.1) | `plots/phq9_distribution.png` | `experiment.ipynb` (savefig cell) | `data/confidential/phq9_filtered.csv` (HELIUS) |
| Tables 1–2 (estimators / shift) | aggregate CSVs + figs | `bash scripts/assessment/run_eval_comparison.sh` → `utils.visualization` | `data/test_post/method_comparison/`, `data/test_post/bert_regression/eval_baseline/aggregate.csv`, `data/test_post/optimized_phq9/*/eval_on_*` |
| `re-infer-loop.pdf`, `LLM_inference`, `Textgrad_diagram`, `tweet_generation_loop`, `BERT-phq9` | — | hand-drawn diagrams | — |

Notes
- The paper's `Fig/` folder is an md5-identical copy of the CS-thesis `Figures/`
  set; 16 of its files are not included by `manuscript.tex` (dead weight there).
- The paper does NOT use: the prompt-variant SA (`prompt_sa*`), the BERT
  fine-tuning figures, or anything from the network simulation (`networks_post`).
- Assessment models: LLM path assesses from posts (+persona in the template);
  BERT+MLP path (`--phq9_mode bert`, used in simulations) assesses from posts
  only — previous PHQ-9 enters only via the opt-in bias correction / cap.
- Manuscript-side TODOs (fix there, not here): empty `\caption{}` at
  `manuscript.tex:96` and `:108`; data/code availability placeholder at `:222`.
