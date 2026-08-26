"""Sensitivity analysis: how much do neighbour and agent sampling move the
output distribution, relative to LLM stochasticity baseline?

Pipeline (3 stages):

  1. ``scripts/sensitivity/sa_run.sh``        — runs 4 settings × 3 replicates per axis (2 axes
                             = 24 generations) via generate_test_data.py.
  2. ``sa_embed.py``      — encodes every post with MentalBERT, saves one
                             ``embeddings.npz`` per run.
  3. ``sa_analyze.py``    — computes within-setting (LLM-noise baseline) and
                             cross-setting (axis effect) cosine similarities,
                             stratifies by PHQ-9 5-band, plots.
"""
