# Exp18 PAI Gate Final Report

Date: 2026-06-18

## Status

```text
PAI_GATE_COMPLETED_NEGATIVE_ABLATION
```

PAI host:

```text
dsw-753014-dc85766cb-4v2jj
```

PAI worktree:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp18_gate
```

Sync strategy:

```text
clean_worktree_plus_hal_git_archive
```

Source commit:

```text
1ff7246 Add Exp18 multi-frame propagation gated DPO
```

## Artifacts

Propagation cache:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/data/cache/exp18_multiframe_propagation_cache_limit100
```

DAVIS10 eval:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp18_multiframe_propagation_gated_dpo_davis10
```

Visual cases:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp18_multiframe_propagation_gated_dpo_davis10/visual_cases/all_methods
```

DPO diagnostics:

```text
exp18_multiframe_propagation_gated_dpo/dpo_diag/
```

## Propagation Cache Quality

The limit=100 propagation cache completed with zero failed samples.

| Metric | Mean | P10 | P50 | P90 |
|---|---:|---:|---:|---:|
| propagation confidence | 0.0376 | 0.0141 | 0.0346 | 0.0613 |
| average source count | 0.2289 | 0.0843 | 0.2441 | 0.3191 |
| propagated-region PSNR | 23.6744 | 17.1210 | 24.0936 | 29.0716 |
| full-mask propagation PSNR | 30.7547 | 22.9774 | 29.4340 | 40.5176 |

Interpretation: the non-oracle propagation signal is real but sparse. It can find some reliable pixels, but the coverage/confidence are not strong enough to dominate the inpainting region.

## DAVIS10 Metrics

| Method | PSNR | SSIM | strict mask PSNR | boundary PSNR | bbox PSNR | bbox SSIM |
|---|---:|---:|---:|---:|---:|---:|
| Exp11 boundary outer b0.75 S2 | 30.2413 | 0.9650 | 18.7114 | 24.8326 | 21.7237 | 0.7603 |
| Exp18a prop-only S1-500 | 30.1024 | 0.9650 | 18.5725 | 24.7090 | 21.5848 | 0.7622 |
| Exp18b prop+gen S1-500 | 29.6892 | 0.9609 | 18.1593 | 24.7152 | 21.1715 | 0.7336 |
| Exp18c oracle S1-500 | 29.7626 | 0.9632 | 18.2326 | 24.7991 | 21.2449 | 0.7490 |
| SFT-48000 baseline | 30.0126 | 0.9635 | 18.4827 | 24.4772 | 21.4950 | 0.7536 |

Best Exp18 variant:

```text
Exp18a prop-only S1-500
```

Metric decision:

```text
No Exp18 variant beats Exp11 outer b0.75 S2 on DAVIS10 primary metrics.
```

## DPO Diagnostic Summary

| Variant | mean loser_dominant | mean prop coverage | mean prop conf | final loss | final dpo_loss | label |
|---|---:|---:|---:|---:|---:|---|
| Exp18a | 0.9412 | 0.0277 | 0.0316 | 0.5519 | 0.4656 | `NON_ORACLE_SPARSE_CONFIDENCE`, `LOSER_DOMINANT` |
| Exp18b | 0.9216 | 0.0228 | 0.0273 | 0.5911 | 0.5170 | `NON_ORACLE_SPARSE_CONFIDENCE`, `NEGATIVE_ABLATION` |
| Exp18c | 0.9608 | 0.9463 | 0.9400 | 0.4954 | 0.4173 | `ORACLE_UPPER_BOUND_NEGATIVE`, `DIAGNOSTIC_ONLY` |

The oracle run is decisive: even with high reliable-region coverage, Exp18c does not beat Exp11. That means this specific loss design is not ready for larger training.

## Visual Judgement

Visual review was performed on DAVIS10 contact sheets covering `boat`, `rhino`, `dog-agility`, `blackswan`, `lucia`, `dance-jump`, `soccerball`, `kite-surf`, `bear`, and `breakdance`.

Result:

```text
No clearly positive Exp18-over-Exp11 case was observed.
```

Exp18a is close to Exp11 in several videos but not better. Exp18b and Exp18c frequently soften local details or introduce small artifacts. `boat` is the clearest negative example: Exp11 preserves water/wake continuity better, while Exp18 variants do not improve the masked water region.

## Decision

Do not continue automatically to:

- full propagation cache
- Stage1 1000
- Stage1 2000
- Stage2

Current best remains:

```text
Exp11 outer b0.75 S2
```

Exp18 should be kept as an exploratory negative ablation. If revisited, the loss formulation should change before any larger run.
