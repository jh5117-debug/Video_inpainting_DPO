# Exp36 MiniMax Winner-SFT Positive-Control

Status: `MINIMAX_POSITIVE_CONTROL_PASS_HELDOUT_QUALITY_NOT_POSITIVE`

This milestone ran a bounded supervised winner-reconstruction positive-control, not DPO and not long training. It was intended to answer whether MiniMax can learn a useful signal from the Gate64 data before any Linear-DPO / SDPO / LocalDPO-inspired rescue.

## Setup

- Runtime root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp36_minimax_objective_rescue/winner_sft_positive_control_20260627_fixed`
- Scopes: `S0`, `S1`
- S0: prior full MiniMax transformer scope.
- S1: Exp36 LoRA attention/projection scope, rank `8`, alpha `16`, dropout `0`.
- Steps per recipe: `10`
- Heldout rows: `4`
- Strict reload: passed for generated recipe checkpoints.
- NaN/Inf: not detected.

## Recipe Metrics

- `S0_adamw_lr0.0001`: loss decrease `0.611931`, delta probe `0.000220452083`, heldout mask delta `-5.053039`, boundary delta `-6.497239`
- `S0_adamw_lr1em05`: loss decrease `0.696504`, delta probe `1.43912518e-05`, heldout mask delta `-0.239675`, boundary delta `-0.657298`
- `S0_adamw_lr3em05`: loss decrease `0.690940`, delta probe `4.17291683e-05`, heldout mask delta `-0.864464`, boundary delta `-2.036330`
- `S1_adamw_lr0.0001`: loss decrease `0.692041`, delta probe `0.000133513061`, heldout mask delta `-0.052968`, boundary delta `-0.202504`
- `S1_adamw_lr1em05`: loss decrease `0.688553`, delta probe `7.67132126e-06`, heldout mask delta `0.000986`, boundary delta `-0.004270`
- `S1_adamw_lr3em05`: loss decrease `0.689080`, delta probe `2.79291808e-05`, heldout mask delta `-0.005604`, boundary delta `-0.030257`

Best technical recipe by local metric was `S1_adamw_lr1em05`, with a tiny heldout mask PSNR delta of `+0.000986` and boundary PSNR delta of `-0.004270`. This is not visually meaningful and does not satisfy quality-positive evidence.

## Codex Visual Review

Codex opened and inspected `24/24` heldout Step0-vs-Step10 temporal strips.

- Better rows: `0`
- Tie / no visible gain rows: `20`
- Clearly worse / new artifact rows: `4`
- Classification counts: `{'CLEARLY_WORSE_NEW_ARTIFACT': 4, 'TIE_METRIC_MIXED': 6, 'TIE_METRIC_WORSE': 6, 'TIE_NO_VISIBLE_CHANGE': 8}`

By recipe:

- `S0_adamw_lr0.0001`: CLEARLY_WORSE_NEW_ARTIFACT=4
- `S0_adamw_lr1em05`: TIE_METRIC_MIXED=1, TIE_METRIC_WORSE=3
- `S0_adamw_lr3em05`: TIE_METRIC_MIXED=1, TIE_METRIC_WORSE=3
- `S1_adamw_lr0.0001`: TIE_METRIC_MIXED=4
- `S1_adamw_lr1em05`: TIE_NO_VISIBLE_CHANGE=4
- `S1_adamw_lr3em05`: TIE_NO_VISIBLE_CHANGE=4

The only recipes that produced obvious visible movement were S0 high-LR updates, and those produced color/texture drift or large local artifacts. S1 LoRA recipes were stable but visually tied with Step0.

## Decision

Technical positive-control passed: MiniMax can reduce supervised flow loss, move parameters, strict-reload checkpoints, and produce non-identical heldout outputs.

Heldout quality-positive control failed: no heldout row was visually better, and the only clear movements were harmful. Therefore Exp36 should not unlock bad-noise mining, objective rescue, 30-step confirmatory training, or third-backbone evidence from this milestone.

Final interpretation: MiniMax failure is not code/loading; it is objective/update-scale/generalization quality. MiniMax remains plumbing-positive, not quality-positive.

## Outputs

- Metrics: `reports/exp36_minimax_winner_sft_metrics.csv`
- Training diagnostics: `reports/exp36_minimax_winner_sft_positive_control.csv`
- Visual review: `reports/exp36_minimax_winner_sft_visual_review.csv`
- Summary: `reports/exp36_minimax_winner_sft_summary.json`
- NAS evidence root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp36_minimax_objective_rescue/winner_sft_positive_control_20260627_fixed`
