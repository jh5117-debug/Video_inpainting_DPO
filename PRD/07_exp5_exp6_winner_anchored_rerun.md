# Exp5 / Exp6 Winner-Anchored Rerun

Updated: 2026-05-31

## Failed Diagnostic Runs

`exp5_d2_comp_k4_stage1/stage2_full` with `beta_dpo=500` is failed /
collapsed / diagnostic only.

`exp5_d2_comp_k4_beta10_s1s2_4000` is also failed / collapsed / diagnostic
only.

The failure is not a task failure, not a Stage2 weight handoff bug, and not a
VBench runner bug. Stage2 loaded Stage1 `last_weights` correctly and completed.
The failure is the unanchored D2 generated-loser + full-mask/full-video DPO
objective.

Observed pattern:

- `mse_w >> ref_mse_w`: policy gets worse on the winner.
- `mse_l >> ref_mse_l`: policy gets worse on the loser.
- `lose_gap` grows more than the winner damage, so the DPO ranking is satisfied.
- `implicit_acc` approaches 1 and `dpo_loss` approaches 0.
- Visual output collapses into universal stripe / high-frequency noise / color
  explosion.

## New Objective

Winner-anchored DPO adds explicit pressure to keep the winner prediction close
to the reference:

```text
L_total = L_DPO
        + lambda_abs * m_w
        + lambda_gap * ReLU(m_w - m_w_ref - margin)
```

Where:

- `m_w` is policy winner MSE.
- `m_w_ref` is reference winner MSE.
- `ReLU(m_w - m_w_ref - margin)` directly suppresses winner-gap explosion.

The default values remain zero, so older experiments keep the old behavior.

## New Exp5

```text
name = exp5_d2_comp_k4_wingap_lose025_beta10_s1s2_4000
manifest = selected_primary_comp.repaired.jsonl
train_mask_mode = full
mask_from_manifest = false
loss_region_mode = full
beta_dpo = 10
winner_abs_reg_weight = 0.05
winner_gap_reg_weight = 1.0
winner_gap_reg_margin = 0.0
lose_gap_weight = 0.25
sft_reg_weight = 0.0
stage1_steps = 4000
stage2_steps = 4000
post_stage2_eval = qual30 + full VBench
```

## New Exp6

```text
name = exp6_d2_nocomp_k4_wingap_lose025_beta10_s1s2_4000
manifest = selected_primary_nocomp.repaired.jsonl
train_mask_mode = full
mask_from_manifest = false
loss_region_mode = full
beta_dpo = 10
winner_abs_reg_weight = 0.05
winner_gap_reg_weight = 1.0
winner_gap_reg_margin = 0.0
lose_gap_weight = 0.25
sft_reg_weight = 0.0
stage1_steps = 4000
stage2_steps = 4000
post_stage2_eval = qual30 + full VBench
```

## H20 Exp6 Policy

All active old Exp6 unanchored training processes should be stopped and marked
superseded. Do not delete old outputs; keep them for failure audit evidence.

Current H20 winner-anchored Exp6:

```text
name = exp6_d2_nocomp_k4_wingap_lose025_beta10_s1s2_4000
status = running / continue
action = monitor only; do not kill
purpose = no-comp data-only comparison against Exp5 comp
```

## Exp7 Partial-Mask Gate

Exp5 winner-anchored is improved but not final. The winner anchor suppressed
the worst unanchored collapse, but the run still shows texture/color attractors
under a data-only full-mask/full-video objective.

Exp7 tests task alignment:

```text
name = exp7_d2_comp_k4_partial_wingap_lose025_beta10_s1s2_gate1500
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
sft_reg_weight = 0.0
stage1_steps = 1500
stage2_steps = 1500
validation = qual30 side-by-side + dpo_diag summary
full_vbench = disabled by default for gate
```

Exp5/6 are still data-only full-mask bridge experiments. Exp7 is a task gate:
training uses the generated loser manifest mask as DiffuEraser's partial-mask
condition. If Exp7 is more stable, the D2 data itself is likely usable and the
main failure was objective/task mismatch.
