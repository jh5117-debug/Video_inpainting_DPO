# Status

- 2026-06-06 10:55 CST: first H20 Exp8c bf16/split run failed at step 0 with
  `SIGFPE`.
- 2026-06-06 11:35 CST: GPU1 one-step smoke passed with fp32/nosplit and wrote
  `dpo_diagnostics.csv`.
- 2026-06-06 11:56 CST: formal fp32/nosplit H20 run reached `global_step=30`
  with `dpo_diagnostics.csv` present.
- 2026-06-06 12:20 CST: same run is still active and reached
  `global_step=100`; no `Traceback`, `ERROR`, `OutOfMemory`, or `SIGFPE`
  was observed in the checked log tail.
- 2026-06-06 12:36 CST: same run is still active and reached
  `global_step=150`; the compact error check is empty.

Current conclusion: running diagnostic only. Do not report final success until
Stage1/Stage2 DAVIS metrics, qualitative videos, and dpo_diag summaries exist.
