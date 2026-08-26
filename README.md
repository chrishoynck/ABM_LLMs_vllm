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
