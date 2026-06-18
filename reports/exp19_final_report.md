# Exp19 Final Report

Date: 2026-06-18

## Status

Exp19b Boundary-Gated Flow-Adapter DPO is no longer blocked by architecture or
inference plumbing. The isolated wrapper, checkpoint loading, DAVIS10 flow
cache, and DAVIS10 evaluation all completed on PAI.

Final gate status:

```text
DAVIS10_EVAL_COMPLETED_NEGATIVE_GATE
```

The result is a safe but almost no-op adaptation relative to Exp11 outer b0.75
S2. It does not justify 1000-step extension, DAVIS50, full cache, or full
training.

## Inference Validation

- flow adapter checkpoint:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/exp19b_boundary_flow_adapter_s2_500_limit100/last_weights/flow_adapter.pt`
- injected modules:
  - `mid_block.motion_modules.0`
  - `up_blocks.0.motion_modules.0`
  - `up_blocks.1.motion_modules.0`
- strict load: yes
- missing keys: none
- unexpected keys: none
- adapter fallback: no
- adapter parameters: `30,723`
- adapter L2 norm: `1.7327`

Preflight:

- status: PASS
- disabled wrapper vs Exp11 MAE: `0.009878`
- enabled vs disabled MAE: `0.009667`
- real-flow vs shuffled-flow MAE: `0.009483`

Interpretation: the external adapter is loaded and the output changes when real
flow context is enabled or shuffled. The disabled wrapper is close to, but not
bit-identical to, the standard Exp11 evaluator under this pipeline path.

## DAVIS10 Flow Cache

- cache path:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/data/cache/exp19_davis10_propainter_completed_flow/`
- videos: `10`
- mean flow confidence: `0.7619`
- mean valid flow ratio: `0.8845`
- mean forward-backward error: `0.5106`

The flow cache is usable for the gate. Confidence was computed from
forward-backward consistency and valid/source-known masks, not GT-error.

## Metrics

Protocol: DAVIS10, raw6, hard comp, no PCM, no mask dilation, no Gaussian blur,
frame-wise metrics. Ewarp used the local RAFT backend. TC was not computed
because the TC backend attempted to download an OpenCLIP checkpoint from
Hugging Face and PAI network access failed.

| method | PSNR | SSIM | LPIPS | Ewarp | strict mask PSNR | boundary PSNR |
|---|---:|---:|---:|---:|---:|---:|
| SFT-48000 | 29.6181 | 0.9620 | 0.02204 | 8.3724 | 18.3203 | 24.2735 |
| Exp11 outer b0.75 S2 | 29.8295 | 0.9633 | 0.02065 | 8.3307 | 18.5317 | 24.6577 |
| Exp19b Stage2-500 | 29.8291 | 0.9633 | 0.02065 | 8.3306 | 18.5313 | 24.6574 |

Delta Exp19b - Exp11:

- PSNR: `-0.00038 dB`
- SSIM: `-0.0000024`
- LPIPS: `-0.0000013`
- Ewarp: `-0.000080`
- strict mask PSNR: `-0.00038 dB`
- boundary PSNR: `-0.00028 dB`

The Ewarp improvement is only about `0.00096%`, far below the required 2%
positive-gate threshold. PSNR, strict mask PSNR, and boundary PSNR are tiny
regressions.

## Visual Review

Reviewed all DAVIS10 contact sheets:

- boat
- rhino
- dog-agility
- blackswan
- lucia
- dance-jump
- flamingo
- soccerball
- camel
- car-roundabout

Exp19b is visually safe: no obvious flow ghosting, double edges, or deformation
were introduced. However, it is also essentially indistinguishable from Exp11.
There is no reliable visible reduction in flicker, moving-boundary artifacts, or
mask/boundary seams.

Better than Exp11: none.

Tie / indistinguishable: boat, rhino, dog-agility, blackswan, lucia,
soccerball, car-roundabout.

Slight negative / no visible benefit with small metric regression: dance-jump,
flamingo, camel.

## Decision

Exp19b does not pass the positive gate:

- TC improvement: unavailable in this run.
- Ewarp relative improvement: far below 2%.
- PSNR / mask / boundary: tiny regressions.
- visual review: no positive temporal signal.

Do not extend to 1000 steps. Do not run DAVIS50. Do not run full-cache or 2000
steps. Keep Exp19 as a completed engineering validation plus negative/neutral
scientific ablation.

Current best remains:

```text
Exp11 outer b0.75 S2
```
