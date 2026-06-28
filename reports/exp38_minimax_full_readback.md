# Exp38 MiniMax Full Adapter Breakthrough Readback

Date: 2026-06-28

Status: `EXP38_READBACK_COMPLETED`

## Git

- Branch: `research/exp38-minimax-full-adapter-breakthrough-20260628`
- Base branch: `origin/research/exp37-minimax-localdpo-badnoise-rescue-20260627`
- Start HEAD: `558c2f263469f4ee6ee46e2a1b26a8082515dded`
- Worktree: `/home/hj/H20_Video_inpainting_DPO_exp38_minimax_full`
- PAI code root target:
  `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp38_minimax_full`
- Output root target:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp38_minimax_full_adapter_breakthrough`
- Log root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp38_minimax_full_adapter_breakthrough`

## Files Read

PRD:

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/50_exp30_vor_or_multimodel_minimax.md`
- `PRD/51_exp35_minimax_flow_dpo_rescue.md`
- `PRD/52_exp36_minimax_objective_rescue.md`
- `PRD/53_exp37_minimax_localdpo_badnoise_rescue.md`
- VideoPainter Exp31 reports were read from
  `origin/research/exp31-videopainter-2000step-longrun-20260627`.
- DiffuEraser evidence references were read from Exp11/Exp15/Exp20 PRD and
  registry files available on this branch.

Registry:

- `experiment_registry/exp30_vor_or_multimodel_minimax/status.md`
- `experiment_registry/exp35_minimax_flow_dpo_rescue/status.md`
- `experiment_registry/exp36_minimax_objective_rescue/status.md`
- `experiment_registry/exp37_minimax_localdpo_badnoise_rescue/status.md`
- Exp37 paths/config/results/metric/qualitative summaries.

Reports:

- `reports/exp30_minimax_gate64_adapter_10step_v3.md`
- `reports/exp30_minimax_gate64_adapter_10step_metrics_v3.csv`
- `reports/exp30_minimax_gate64_adapter_10step_visual_review_v3.csv`
- `reports/exp35_minimax_rescue_10step_summary.json`
- `reports/exp36_minimax_inference_sensitivity_summary.json`
- `reports/exp36_minimax_winner_sft_summary.json`
- `reports/exp37_minimax_localdpo_badnoise_10step_summary.json`
- `reports/exp37_minimax_paper_positioning.md`
- Exp31 VideoPainter 2000 final decision, LPIPS/Ewarp metrics, paper evidence,
  and visual review reports from the Exp31 branch.

Code:

- Exp30 MiniMax Gate64 adapter gate and OR source-pool scripts.
- Exp35 MiniMax no-change, sensitivity, bad-noise, and rescue scripts.
- Exp36 MiniMax sensitivity, LoRA scope, and winner-SFT scripts.
- Exp37 LocalDPO-style corruption, bad-noise miner, train-vs-heldout, and
  LocalDPO-badnoise 10-step runner.

## 1. What Failed In Exp30 / Exp35 / Exp36 / Exp37?

Exp30: Gate64 MiniMax frozen and EMA Linear-DPO recipes passed zero-gap and
one-step strict reload but produced no heldout visual improvement. Visual
better count was `0/32`; mean heldout mask and boundary PSNR deltas were near
zero or slightly negative.

Exp35: hard-noise Linear-DPO rescue recipes produced nonzero movement but no
quality-positive heldout result. Codex reviewed `48/48` heldout strips and
found `0` visually better rows. Mean mask, boundary, and outside PSNR deltas
were negative for all R1/R2/R3 recipes.

Exp36: checkpoint/load and ignored-weight hypotheses were ruled out. MiniMax
inference consumed trained weights, and winner-SFT lowered train loss and moved
outputs. However, heldout quality still did not improve: `0/24` visual better
rows, with high LR introducing artifacts.

Exp37: cleaner LocalDPO-style corruption data and outside-sane bad-noise states
were built, but the preregistered 10-step rescue was only Pareto mixed. R1
improved full/mask/outside PSNR on average but missed boundary PSNR and had
only `1/16` visually better rows. R2/R3 degraded metrics and also had only
`1/16` visually better rows.

## 2. What Was Ruled Out?

- Checkpoint fallback is not supported.
- Inference ignoring trained weights is ruled out by Exp36 sensitivity:
  identity replay max MAE `0.0`, and 1.01x perturbation mean full/mask MAE
  `0.088218` / `0.156302`.
- Total inability to train is ruled out: winner-SFT reduces train loss and
  moves outputs.
- Small adapter scope is not the sole explanation: Exp30/35 used full
  transformer scope, and Exp36 prepared S1 LoRA scope but still lacked heldout
  quality.
- Collapse is not the dominant failure: most failed recipes were no-change,
  tie, or slight degradation, not black/purple collapse.

## 3. What Remains Unresolved?

- Whether official MiniMax reproduction settings, preprocessing, prompt/task
  format, timestep sampling, or scheduler choices differ from the adapter
  wrapper enough to explain weak transfer.
- Whether the objective is targeting low-amplitude texture shifts rather than
  visible local removal defects.
- Whether the current data pool is too broad or still not aligned with
  MiniMax's native failure modes.
- Whether stronger positive controls can overfit a tiny set and visibly improve
  the same training videos before any heldout rescue.
- Whether region weighting, utility normalization, hard-noise selection, or
  update scale must be redesigned before 10-step DPO.

## 4. Recipes Not To Repeat

Do not repeat:

- Exp30 frozen/EMA 10-step Linear-DPO on Gate64 V3.
- Exp35 R1/R2/R3 hard-noise Linear-DPO recipes.
- Exp36 naive winner-SFT as a quality claim.
- Exp37 R1/R2/R3 LocalDPO-badnoise recipes at the same utility scale, LR, and
  hard-state settings.

Any new recipe must first be justified by failure taxonomy, official protocol
audit, train-overfit diagnosis, and preregistration.

## 5. Data Pools Ready

- Exp30 Gate64 V3 train32 / heldout16 pool is ready and scene-disjoint.
- Exp37 LocalDPO-style corruption train32 / heldout16 pool is ready, visually
  reviewed `48/48`, with `38` medium-hard and `10` hard-plausible final rows.
- Exp37 bad-noise states are ready: train32 x 64 candidate states, manifest
  SHA256 `492210b2cd725faa348adcbafaf37bf82cc6790b4eb0607b9f758047d1c795d4`.

## 6. Trainable Scopes Ready

- S0: full MiniMax transformer scope from Exp30/35.
- S1: Exp36 LoRA attention/projection rank8 alpha16 scope prepared and tested
  structurally.
- S2: last-four-block MLP LoRA remains locked until S1 or another diagnostic
  shows useful positive-control evidence.

## 7. Objectives With Some Promise

- Exp37 R1 `LocalDPO-Linear-HardNoise` is the only recipe with mixed positive
  numeric movement: full/mask/outside PSNR deltas `+0.200826` / `+0.161946` /
  `+0.028198`, but boundary PSNR `-0.049755` and only `1/16` visible
  improvement.
- SDPO-safe and SFT-warmup hybrids did not help in Exp37; both were metric
  negative.
- Winner-SFT is useful as a diagnostic positive-control, not as paper evidence
  or direct rescue.

## 8. Why MiniMax Is Not Abandoned

MiniMax remains worth debugging because:

- official repo and weights are available;
- it is a distinct flow-style Wan/DiT architecture;
- flow target wiring is known as `epsilon - z0`;
- zero-gap, one-step strict reload, and sensitivity checks passed;
- training can move outputs;
- LocalDPO-style clean data and bad-noise states now exist.

It is not paper-positive yet because no heldout visual-quality gate has passed.

## 9. Planned Exploration Schedule

1. Failure taxonomy and decision tree.
2. Train-overfit diagnosis on the strongest available controls.
3. Official MiniMax reproduction/protocol audit.
4. LocalDPO pool v2 only if the taxonomy indicates data mismatch.
5. Bad-noise v2 only if official protocol/timestep audit justifies it.
6. SFT warmup ladder with strict visual and metric gates.
7. Bounded DPO preregistration.
8. 10-step quality-positive gate.
9. Conditional 30-step only after a strict 10-step pass.
10. Conditional 100/300/500-step only after smaller positive gates and explicit
    user authorization.

## 10. GPU Availability

PAI host: `dsw-753014-85f54df947-bkp7h`.

Two readback checks showed:

| GPU | State |
| --- | --- |
| 0 | 0 MiB, 0%, no compute PID |
| 1 | 0 MiB, 0%, no compute PID; legacy `runtime/cli4` GPU1 lock files exist |
| 2 | occupied, `/usr/local/bin/python3.10`, about 136 GiB, 100% |
| 3 | occupied, `/usr/local/bin/python3.10`, about 136 GiB, 100% |
| 4 | occupied, `/usr/local/bin/python3.10`, about 136 GiB, 100% |
| 5 | occupied, `/usr/local/bin/python3.10`, about 136 GiB, 100% |
| 6 | occupied, `/usr/local/bin/python3.10`, about 136 GiB, 100% |
| 7 | occupied, `/usr/local/bin/python3.10`, about 136 GiB, 100% |

No signal was sent and no unknown task was modified. GPU work was not launched
in this readback.

## 11. Permission State

PAI `hj` can write:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch
```

and the Exp38 log directory was created.

PAI `hj` cannot currently write:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo
```

Future checkpoint/output milestones need a minimal root permission fix for the
Exp38 experiment directory, or they must use the writable log root for
non-checkpoint diagnostic outputs until fixed. No broad NAS permission changes
were attempted.

## 12. Paper Evidence To Collect

DiffuEraser:

- Exp11 outer b0.75 S2 DAVIS50 / YouTubeVOS100 metrics and final visual cases.
- Exp15 OR DAVIS50 background-preservation benchmark for OR context.
- Exp20 negative boundary search to document what did not generalize.

VideoPainter:

- Exp26 shadow-dev confirmation and result pack.
- Exp31 2000-step final decision, LPIPS/Ewarp metrics, search/shadow visual
  reviews, and paper evidence table.

MiniMax:

- Exp30/35/36/37 negative and plumbing-positive evidence.
- Exp38 diagnostics and any future positive gates, if achieved.

Allowed paper claim today:

```text
DiffuEraser + VideoPainter provide cross-backbone adapter evidence;
MiniMax remains plumbing-positive but unresolved.
```

Forbidden:

```text
UNIVERSAL_ADAPTER
ALL_MODELS_SUPPORTED
FINAL_SOTA
TOP_CONFERENCE_NOVELTY_CONFIRMED
MINIMAX_THIRD_BACKBONE_SUCCESS
```

