# Exp50 Metric Summary

Last updated: 2026-06-30T16:44:02+08:00

No training or inference metrics yet.

Weight relay:
- VOID files: 12
- Base files: 40
- SHA match: yes

Environment smoke:
- Status: `VOID_ENV_PARTIAL`
- Imports pass: 44
- CUDA failures: 0
- Version pin warnings: 8

VOR-to-VOID Gate8:
- Status: `VOID_VOR_QUADMASK_GATE8_READY`
- Gate rows: 8
- REAL / BLENDER: 4 / 4
- Small / medium / large: 3 / 3 / 2
- Scene overlap: False
- VOR-Eval excluded: True

## C2 Environment Repair (2026-06-30T15:25:36+08:00)

No inference metrics were produced. `VOID_ENV_READY` was not reached due to `VOID_ENV_BLOCKED_TORCH` and `VOID_ENV_BLOCKED_DEEPSPEED`.

## C3 Environment Relay Ingest (2026-06-30T16:44:02+08:00)

- Status: `VOID_ENV_READY`
- Wheelhouse files: 145 (3.3G)
- Transfer hash match: True; missing 0; mismatches 0
- Env torch: `2.7.1+cu126`; CUDA runtime `12.6`
- Import failures: 0
- CUDA bf16 tiny smoke: `PENDING_NO_FREE_GPU`; max allocation 67146240 bytes

## F0 Component Load Smoke (2026-06-30T16:50:03+08:00)

- Status: `VOID_COMPONENT_LOAD_PASS`
- Pass1 keys: 1024; Pass2 keys: 1024
- Base transformer keys: 1024; Base VAE keys: 436
- Total asset disk bytes: 43385161698
- Full GPU model load: not attempted

## F1 Official Sample Inference (2026-06-30T17:02:21+08:00)

- Status: `VOID_OFFICIAL_SAMPLE_INFERENCE_PASS`
- Sample: `lime`; return code `0`; raw frame count 85
- Raw outputs: 1; tuple outputs: 1
- Checkpoint load clean: True; sampling 30/30: True

## F2 VOR Gate8 Inference Metrics - 2026-06-30T17:24:15+08:00

- Status: `VOID_INFERENCE_SMOKE_PASS`
- Technical valid: 8/8
- Mean PSNR: 30.1749
- Mean SSIM: 0.8244
- Mean mask PSNR: 25.5380
- Mean boundary PSNR: 25.8435
- Mean outside PSNR: 33.1091
- Mean outside L1: 4.3402
- LPIPS/Ewarp/TC unavailable in this smoke; not used for promotion.

## G0 Adapter Micro Data - 2026-06-30T17:29:15+08:00

- Status: `VOID_ADAPTER_MICRO_DATA_READY`
- Train/Heldout: 4 / 4
- Loser sources: {'controlled_local_corruption': 6, 'void_pass1_raw_medium_hard': 2}
- Metrics were not recomputed in G0; this is data preparation only.

## G1 Zero-Gap / One-Step - 2026-06-30T17:37:17+08:00

- Status: `VOID_TRAINABLE_FORWARD_BLOCKED`
- Dataset load: PASS in official bucket mode.
- Zero-gap metrics: not run, blocked before model preference forward.
- One-step metrics: not run, blocked before optimizer.
