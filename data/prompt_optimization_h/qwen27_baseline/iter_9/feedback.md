# Iter 9 — qwen27_baseline

model:    Qwen/Qwen3.5-27B
agents:   7  (sample seed = 9)

## What you saw
<read posts.csv; jot down patterns, repetition, off-tone outputs, etc.>
Good extreme tweets, but the lower PHQ-9 was way to heavy
## What to try next
<concrete edits for iter_10/prompt.txt>
Go back to iter 7, and reduce the extremety of lower PHQ-9. 

## Scores (fill in by hand)
training_score: 7.7
scores: 9, 9, 7, 10, 9, 3, 7
val_score: 7.05 ± 1.80  (teacher, 20 held-out personas, val20/)
