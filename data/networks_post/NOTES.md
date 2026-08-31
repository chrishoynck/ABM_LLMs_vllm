# data/networks_post/ — notes (cleanup 2026-08)

Simulation run trees (GABM thesis). Layout via `PathManager`
(src/utils/tools/path_manager.py): `{basis,happy}/{sda,sdc}/{directed,undirected}/
{debiased,non_debiased}/<config>/rounds<R>_N<A>/seed_<S>/`.

- `rounds300_N100/` = finished runs (all drivers use ROUNDS=300); intermediate
  checkpoints are not kept, so `--use_saved_network` can only resume finished
  runs (no live workflow resumes unfinished ones).
- `basis/sda/undirected/old_debiased/` — KEPT despite the name: sole consumer is
  `plot_network_targets.py:88` (`_K6_RUNS` k=6 reference,
  `old_debiased/2_1655_d6_dim5/rounds300_N100/seed_*/meta.json`).
- The EXCLUDE lists (run_plot_evolution.sh:58, plot_network_evolution.py:693)
  name `old_pop` / `old_debiased` / `different_debias_settings`; only
  `old_debiased` still matches anything — harmless, left as-is.
- Config-dir naming: decimal degrees are `_`-escaped (`2_1655_d4_5_dim5` =
  α 2.1655, degree 4.5, dim 5) — beware `d8_2539` reads as degree 8.2539.
