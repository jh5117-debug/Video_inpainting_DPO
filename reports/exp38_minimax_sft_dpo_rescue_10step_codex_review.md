# Exp38 MiniMax SFT/DPO Rescue 10-Step Codex Review

Status: `MINIMAX_SFT_DPO_RESCUE_10STEP_NEGATIVE`.

The bounded GPU1 rescue completed for R1/R2/R3 on the filtered LocalDPO v2 heldout13 set. GPU0 and GPU1 were audited before and after; no GPU0/1 processes needed to be killed, and GPU2-4 jobs were left untouched.

Visual review note: Codex opened the generated full montage and representative individual high-diff/ambiguous 16-frame temporal strips. Because no recipe produced a clear positive result, this report does not promote `VIDEO_REVIEW_PASS`; it records a conservative no-pass decision from temporal-strip evidence.

## Aggregate Recipe Metrics

| recipe | name | full PSNR delta | mask PSNR delta | boundary PSNR delta | outside PSNR delta | mean pixel diff | visual better | visual worse/tradeoff | decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| R1 | LocalDPO-Linear-HardNoise | +0.102167 | +0.117230 | -0.141510 | -0.037262 | 2.845801 | 0/13 | 9/13 | best numeric signal, but not quality-positive |
| R2 | LocalDPO-Linear-SDPO | -0.258482 | -0.078807 | -0.475071 | -0.698459 | 4.028547 | 0/13 | 11/13 | negative |
| R3 | LocalDPO-SFTWarmup-Linear | -0.604098 | -0.159184 | -0.668335 | -1.528854 | 6.057883 | 0/13 | 12/13 | negative |

## Findings

- R1 (`LocalDPO-Linear-HardNoise`) is the only recipe with positive mean full/mask PSNR (`+0.102167` / `+0.117230`), but it loses boundary PSNR (`-0.141510`) and slightly hurts outside PSNR (`-0.037262`). The clearest high-delta strip over-erases/softens local structure instead of producing a reliable quality gain.
- R2 (`LocalDPO-Linear-SDPO`) has SDPO preflight coverage (`lambda_mean=0.927594`) but worsens full, mask, boundary, and outside aggregates; it is rejected.
- R3 (`LocalDPO-SFTWarmup-Linear`) moves outputs the most, but this is mostly harmful drift: full `-0.604098`, boundary `-0.668335`, outside `-1.528854`. It is rejected.
- No recipe satisfies the preregistered gate: heldout better >= 6/16, at least two local/effect metrics improved, outside not systematically worse, and Step10 not merely tie/no-change.
- 30-step remains locked. There is no MiniMax third-backbone quality-positive evidence from this rescue.

## Artifacts

- PAI output root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp38_minimax_full_adapter_breakthrough/sft_dpo_rescue_20260628`
- Local montage: `reports/exp38_sft_dpo_rescue_runtime/strips_montage.jpg`
- Reviewed CSV: `reports/exp38_minimax_sft_dpo_rescue_10step_visual_review.csv`
- Metrics CSV: `reports/exp38_minimax_sft_dpo_rescue_10step_metrics.csv`
- Diagnostics CSV: `reports/exp38_minimax_sft_dpo_rescue_10step_diagnostics.csv`
- Summary JSON: `reports/exp38_minimax_sft_dpo_rescue_10step_summary.json`
