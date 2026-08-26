# data/networks_post/ — notes (cleanup 2026-08)

Simulation run trees (GABM thesis). Layout via `PathManager`
(src/utils/tools/path_manager.py): `{basis,happy}/{sda,sdc}/{directed,undirected}/
{debiased,non_debiased}/<config>/rounds<R>_N<A>/seed_<S>/`.

- `rounds300_N100/` = finished runs (all drivers use ROUNDS=300).
- Partial checkpoints `rounds{20..280}_N100/` were binned →
  `bin/data/networks_post/...`. Caveat: resuming an UNFINISHED run with
  `--use_saved_network` would need its checkpoint back (`mv` from bin/);
  no live workflow does this.
- `basis/sda/undirected/old_debiased/` — KEPT despite the name: sole consumer is
  `plot_network_targets.py:88` (`_K6_RUNS` k=6 reference,
  `old_debiased/2_1655_d6_dim5/rounds300_N100/seed_*/meta.json`).
- `old_pop/` and `wrong_run/` binned. `wrong_run` was NOT in the plot-scan
  EXCLUDE lists, so scans silently walked it — the move fixed that.
- The EXCLUDE lists (run_plot_evolution.sh:58, plot_network_evolution.py:693)
  still name `old_pop` / `old_debiased` / `different_debias_settings`; the first
  and last now match nothing — harmless, left as-is (no code rewrites).
- Config-dir naming: decimal degrees are `_`-escaped (`2_1655_d4_5_dim5` =
  α 2.1655, degree 4.5, dim 5) — beware `d8_2539` reads as degree 8.2539.
