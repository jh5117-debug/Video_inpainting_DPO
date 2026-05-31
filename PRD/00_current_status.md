# Current Status

Updated: 2026-05-30

## D2 Data Readiness

The D2 generated-loser dataset is ready and must not be regenerated for the
beta10 reruns.

Root:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4
```

Ready manifests:

| Manifest | Rows |
| --- | ---: |
| `selected_primary_comp.repaired.jsonl` | 10000 |
| `selected_primary_nocomp.repaired.jsonl` | 10000 |
| `selected_secondary_comp.repaired.jsonl` | 10000 |
| `selected_secondary_nocomp.repaired.jsonl` | 10000 |

## Exp5 Collapse Status

`exp5_d2_comp_k4_stage1/stage2_full` with `beta_dpo=500` and 10000-step
Stage1/Stage2 is marked **failed / collapsed / diagnostic only**.

Evidence:

- Stage2 10000 full-mask VBench qualitative outputs show visual collapse.
- Side-by-side videos show the exp5 side as high-frequency noise and color
  explosion rather than coherent inpainting.
- DPO diagnostics saturated early with `acc=1`, `dpo=0`, and `loss=0`;
  this indicates preference-objective saturation, not image quality.
- VBench showed weak downstream behavior, including `dynamic_degree=0`,
  low `overall_consistency`, and poor scene/spatial/object dimensions.

Interpretation:

- This is not a task or code-path failure.
- Exp3 showed the VideoDPO-to-DiffuEraser DPO bridge can work.
- Exp5 failed because D2 generated losers plus full-mask training, full-video
  loss, `beta_dpo=500`, no SFT regularization, and long 10000-step training
  over-optimized the preference signal and pushed the model off distribution.
- Old Exp5 beta500 must not be used as a final result. Keep it only as a
  failed ablation and diagnostic artifact.

`exp5_d2_comp_k4_beta10_s1s2_4000` is also marked **failed / collapsed /
diagnostic only**. Its Stage2 correctly loaded Stage1 `last_weights` and
completed 4000 steps, but the visual outputs collapsed into prompt-insensitive
stripe/high-frequency textures. The log shows the same degenerate mechanism:

- `mse_w >> ref_mse_w`: the policy got worse on the winner.
- `mse_l >> ref_mse_l`: the policy got even worse on the loser.
- `win_gap` and `lose_gap` both grew, while `implicit_acc` stayed near 1.
- `dpo_loss` often approached 0, so the ranking objective was satisfied even
  though visual quality collapsed.

This confirms the failure is not a Stage2 handoff bug and not a VBench script
bug. It is a degenerate solution of D2 generated-loser + full-mask/full-video
DPO: ranking can be improved by making the loser worse while also damaging the
winner.

Replacement:

| Experiment | Status | beta_dpo | Stage1 | Stage2 | Eval |
| --- | --- | ---: | ---: | ---: | --- |
| `exp5_d2_comp_k4_wingap_lose025_beta10_s1s2_4000` | planned/running | 10 | 4000 | 4000 | qual30 + full VBench |

## H20 Old Exp6 beta500 Status

All old H20 Exp6 training using the unanchored objective is superseded and
should be stopped if still running.

Reason:

- It has the same collapse risk exposed by Exp5 beta500 and Exp5 beta10.
- It should not be continued as a final result.

Replacement:

| Experiment | Status | beta_dpo | Stage1 | Stage2 | Eval |
| --- | --- | ---: | ---: | ---: | --- |
| `exp6_d2_nocomp_k4_wingap_lose025_beta10_s1s2_4000` | planned/running | 10 | 4000 | 4000 | qual30 + full VBench |

## Current Run Policy

- Do not continue old beta500 Exp5/Exp6 or unanchored beta10 Exp5/Exp6 long
  training.
- Do not regenerate D2.
- Do not restore D1 full-mask work.
- Do not touch Exp8 region-loss settings in this pass.
- New reruns use winner-anchored DPO with `beta_dpo=10`,
  `lose_gap_weight=0.25`, `winner_abs_reg_weight=0.05`,
  `winner_gap_reg_weight=1.0`, 4000 Stage1 steps, 4000 Stage2 steps, no
  validation during training, then automatic qual30 and full VBench.

## 2026-05-31 Exp7 Partial-Mask Gate

`exp5_d2_comp_k4_wingap_lose025_beta10_s1s2_4000` is improved relative to the
unanchored Exp5 runs: winner-gap regularization suppresses the universal blue
stripe/high-frequency collapse mode. It is not yet a final success because the
data-only full-mask/full-video objective still shows texture and color
attractors in qualitative side-by-side videos.

H20 `exp6_d2_nocomp_k4_wingap_lose025_beta10_s1s2_4000` is running and must
continue. Action: monitor only; do not kill. Purpose: no-comp data-only
comparison against Exp5 comp.

Next gate:

```text
EXP_NAME = exp7_d2_comp_k4_partial_wingap_lose025_beta10_s1s2_gate1500
status = planned / launching on PAI
purpose = test whether aligning the training task with D2 partial-mask loser
          generation stabilizes training
manifest = selected_primary_comp.repaired.jsonl
train_mask_mode = partial
mask_from_manifest = true
M_train = M_gen from manifest mask_path
loss_region_mode = full
beta_dpo = 10
winner_abs_reg_weight = 0.05
winner_gap_reg_weight = 1.0
winner_gap_reg_margin = 0.0
lose_gap_weight = 0.25
stage1_steps = 1500
stage2_steps = 1500
validation = qual30 side-by-side + dpo_diag summary
full_vbench = disabled by default for gate
```

Exp5/6 remain data-only full-mask bridge experiments. Exp7 is a task-alignment
gate: DiffuEraser receives the same partial mask used to generate the D2 loser.
If Exp7 is more stable than Exp5, D2 is not inherently bad; the main issue is
the data-only full-mask objective mismatch.
