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

## H0 Preference-Wrapper Readback - 2026-06-30T22:59:39+08:00

- Status: `VOID_PREFERENCE_WRAPPER_REQUIRED_CONFIRMED`
- Metrics: not applicable; readback only.
- Target parameterization: scheduler `epsilon` or `v_prediction`, read at runtime by wrapper.

## H1 SFT Forward Parity - 2026-06-30T23:08:07+08:00

- Status: `VOID_SFT_FORWARD_PARITY_EXPLAINED`
- SFT loss: 0.035282157361507416
- Target parameterization: `v_prediction`
- Shapes: latent [1, 4, 16, 48, 84]; inpaint [1, 4, 32, 48, 84]

## H2 Preference Forward - 2026-06-30T23:15:48+08:00

- Status: `VOID_PREFERENCE_FORWARD_PASS`
- Winner policy/reference loss: 0.0640571117401123 / 0.0640571117401123
- Loser policy/reference loss: 0.08385218679904938 / 0.08385218679904938
- DPO loss: 0.6931471824645996
- Grad norm: 0.011950799647202132

## H3 Zero-Gap V2 - 2026-06-30T23:17:43+08:00

- Status: `VOID_ZERO_GAP_PASS`
- Winner/loser gaps: 0.0 / 0.0
- DPO loss: 0.6931471824645996; expected log(2): 0.6931471805599453

## H4 One-Step V2 - 2026-06-30T23:24:57+08:00

- Status: `VOID_ONE_STEP_PARETO_MIXED`
- Loss before step: 0.6931471824645996
- Grad norm: 0.011962890625
- Max param delta norm: 0.005055009387433529
- Step1 vs Step0 L1: 0.019980037584900856


## H6 Preference Wrapper Decision - 2026-06-30T23:30:58+08:00

- Status: `VOID_TRUE_ADAPTER_FEASIBLE_NEEDS_MICRO_TRAINING`
- Preference forward: `VOID_PREFERENCE_FORWARD_PASS`; DPO loss 0.6931471824645996; grad norm 0.011950799647202132
- Zero-gap: `VOID_ZERO_GAP_PASS`; winner/loser gaps 0.0 / 0.0; DPO 0.6931471824645996
- One-step: `VOID_ONE_STEP_PARETO_MIXED`; param delta 0.005055009387433529; heldout forward finite True; video metrics not generated
- 10-step: not run


## H4b-0 One-Step Evidence Readback - 2026-07-01T00:05:40+08:00

- Status: `VOID_ONE_STEP_EVIDENCE_READBACK_DONE`
- One-step checkpoint exists for later audit: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp50_pai_void_adapter_feasibility/one_step_v2/adapter_proj_out_step1.pt`
- No new metrics; readback only.


## H4b-1 One-Step Checkpoint Audit - 2026-07-01T00:08:01+08:00

- Status: `VOID_ONE_STEP_CHECKPOINT_READY`
- Checkpoint bytes: 788893
- SHA256: `849326121699e51673990df7bef52c578245812ad934a88bbe6f2acf93b972d5`
- Adapter keys: ['proj_out.bias', 'proj_out.weight']
- No video metrics; checkpoint audit only.


## H4b-2 One-Step Heldout Generation - 2026-07-01T00:09:39+08:00

- Status: `VOID_ONE_STEP_HELDOUT_GENERATION_BLOCKED`
- No Step0/Step1 video metrics generated because all PAI GPUs were occupied.
