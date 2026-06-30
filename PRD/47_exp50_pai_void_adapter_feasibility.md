# Exp50 PAI VOID Adapter Feasibility

Date: 2026-06-30

Branch: `research/exp50-pai-void-adapter-feasibility-20260630`

Status: `VOID_OFFICIAL_SAMPLE_INFERENCE_PASS`

## Objective

Evaluate VOID as a possible third video-inpainting adapter candidate after ROSE was paused. VOID is prioritized because public sources expose code, model checkpoints, data-generation code, and explicit training scripts.

## Safety Boundaries

- Do not continue ROSE downloads or training.
- Do not modify `inference/metrics.py`.
- Do not modify shared trainer code.
- Do not modify official VOID source.
- Do not use VOR-Eval for training, filtering, or threshold design.
- Do not use hard comp for promotion.
- Do not run 50/100/300/500/1000/2000-step training.
- Do not claim VOID third-backbone positive unless one-step and heldout 10-step micro gates truly pass.

## Paths

- HAL worktree target: `/home/hj/H20_Video_inpainting_DPO_exp50_void_adapter`
- PAI worktree fallback in use: `/home/hj/H20_Video_inpainting_DPO_exp50_void_adapter`
- Requested primary PAI worktree: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp50_void_adapter` (not writable by `hj` at readback)
- Output root: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp50_pai_void_adapter_feasibility`
- Log root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp50_pai_void_adapter_feasibility`
- Runtime root: `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp50_pai_void_adapter_feasibility`
- Third-party root: `/mnt/nas/hj/H20_Video_inpainting_DPO/third_party/VOID`
- Weights root: `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/void`
- Data root: `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/void`

## Milestone A Readback

Generated:

- `reports/exp50_void_readback.md`
- `reports/exp50_void_public_source_audit.md`
- `reports/exp50_void_public_source_audit.csv`
- `reports/exp50_void_readback_summary.json`

Key result: VOID has stronger adapter feasibility than ROSE because its public repo documents `data_generation/`, `scripts/cogvideox_fun/train_void.sh`, `scripts/cogvideox_fun/train_void_warped_noise.sh`, quadmask conditioning, and Pass1/Pass2 checkpoints. No download was performed before this readback commit.

## Permission Caveat

At readback, `hj` can write logs/runtime but cannot create asset/output directories under several requested NAS roots. Milestone B must first resolve directory ownership or use an explicitly approved fallback.


## Milestone B Update - 2026-06-30

Status: `VOID_ASSETS_BLOCKED`.

Pre-download disk and permission checks were run. No VOID official repo, Pass1/Pass2 weights, CogVideoX-Fun base model, or sample data was downloaded because the required NAS target roots are not writable by `hj`, and neither passwordless sudo nor root SSH is available from this session. The blocked target roots are `/third_party/VOID`, `/weights/void`, `/data/external/void`, and `/experiments/dpo/exp50_pai_void_adapter_feasibility` under `/mnt/nas/hj/H20_Video_inpainting_DPO`.

The task is blocked until those directories are created/chowned for `hj`, or an explicit alternate asset/output root is approved.


## Milestone B0 Update - 2026-06-30

Status: `VOID_ASSET_PERMISSION_RECOVERED`.

Using `hj` identity, Codex verified read/write/execute access and write-probe behavior for all six EXP50 directories after the user-provided root-side minimal permission fix. Codex did not run chmod/chown or any root permission script.

Reports:

- `reports/exp50_void_permission_recovery.md`
- `reports/exp50_void_permission_recovery.csv`


## Milestone B1 Update - 2026-06-30

Status: `VOID_REPO_READY`.

The official VOID repository was cloned/audited at `/mnt/nas/hj/H20_Video_inpainting_DPO/third_party/VOID/Netflix_void-model`. Reports:

- `reports/exp50_void_repo_audit.md`
- `reports/exp50_void_repo_audit.csv`
- `reports/exp50_void_repo_audit_summary.json`
- `reports/exp50_void_repo_inventory.txt`

No official VOID source was modified.


## Milestone B2 Update - 2026-06-30

Status: `VOID_WEIGHT_DOWNLOAD_BLOCKED`.

Official HuggingFace downloads were attempted for `netflix/void-model` Pass1/Pass2 and `alibaba-pai/CogVideoX-Fun-V1.5-5b-InP`. Both failed from PAI with `httpx.ConnectError: [Errno 101] Network is unreachable`. No fallback, mirror, or fabricated asset was used. Env smoke, trainable-forward audit, quadmask Gate8, and inference smoke remain blocked until the exact weights/base model are available.

Reports:

- `reports/exp50_void_weight_download.md`
- `reports/exp50_void_weight_download.csv`
- `reports/exp50_void_weight_inventory.txt`
- `reports/exp50_void_weight_sha256.txt`
- `reports/exp50_void_asset_download_summary.json`

## Milestone B3 update - VOID_WEIGHTS_READY

- Time: 2026-06-30T14:04:06.075339+08:00
- Status: `VOID_WEIGHTS_READY`
- Evidence: `reports/exp50_void_weight_relay_ingest.md`
- Relay SHA match: yes, 52 / 52 files, missing 0, mismatch 0.
- Safety: no training, no inference, no GPU, no VOID positive claim.

## Milestone C update - VOID_ENV_PARTIAL

- Time: 2026-06-30T14:20:58.202107+08:00
- Status: `VOID_ENV_PARTIAL`
- Evidence: `reports/exp50_void_env_smoke.md`
- Imports: 44 pass, 0 fail.
- CUDA smoke: no failures; small matmul/backward finite.
- Caveat: exact official pins are not matched; heavyweight CUDA packages were not reinstalled.
- Safety: no training, no inference, no full 5B model load, no VOID positive claim.

## Milestone D update - VOID_TRAINING_FORWARD_HEAVY_BUT_POSSIBLE

- Time: 2026-06-30T14:24:40.037508+08:00
- Status: `VOID_TRAINING_FORWARD_HEAVY_BUT_POSSIBLE`
- Evidence: `reports/exp50_void_trainable_forward_audit.md`
- Finding: official trainable forward exists, but default training is heavy transformer fine-tuning, not out-of-box LoVI-DPO.
- Safety: no training, no inference, no official source modification, no VOID positive claim.

## Milestone E update - VOID_VOR_QUADMASK_GATE8_READY

- Time: 2026-06-30T14:40:28.047764+08:00
- Status: `VOID_VOR_QUADMASK_GATE8_READY`
- Evidence: `reports/exp50_void_vor_quadmask_adapter.md` and `reports/exp50_void_vor_quadmask_visual_review.csv`
- Gate8: 8 rows, REAL/BLENDER 4/4, scene overlap False.
- VOR-Eval excluded: True
- Safety: no training, no inference, no hard comp, no VOID positive claim.
- Next gate: official inference smoke remains blocked until environment status is `VOID_ENV_READY`; current C status is `VOID_ENV_PARTIAL`.

<!-- EXP50_C2_ENV_REPAIR -->

### Exp50 C2 VOID Environment Repair - 2026-06-30T15:25:36+08:00

- Status: `VOID_ENV_PARTIAL`.
- Exact blockers: `VOID_ENV_BLOCKED_TORCH`, `VOID_ENV_BLOCKED_DEEPSPEED`.
- `VOID_ENV_READY` not reached; F0/F1/F2/G gates not run.
- Safety: no inference, no training, no optimizer step, no VOID official source modification.

## Milestone C3 update - VOID_ENV_READY

- Time: 2026-06-30T16:44:02+08:00
- Status: `VOID_ENV_READY`
- Evidence: `reports/exp50_void_env_relay_ingest.md`
- Env: `/home/hj/conda_envs/void_exp50_official_v2`
- Torch: `2.7.1+cu126`; CUDA runtime `12.6`; bf16 supported `True`
- CUDA tiny smoke: `CUDA_BF16_BACKWARD_OK` on GPU `0`; max allocation 67146240 bytes.
- Wheelhouse transfer: `VOID_ENV_WHEELHOUSE_TRANSFER_VERIFIED`; hash match True; missing 0; mismatches 0.
- Deepspeed caveat: `DEEPSPEED_TRAIN_ONLY_NOT_INSTALLED_NO_DEPS_SDIST_AVAILABLE`; intentionally not installed to avoid wrong torch/CUDA drift.
- Safety: no training, no inference, no optimizer step, no PAI base env modification, no VOID official source modification, no VOID positive claim.
- Next gate: F0 component load smoke.

## Milestone F0 update - VOID_COMPONENT_LOAD_PASS

- Time: 2026-06-30T16:50:03+08:00
- Status: `VOID_COMPONENT_LOAD_PASS`
- Evidence: `reports/exp50_void_component_load_smoke.md`
- Config/tokenizer/scheduler metadata loaded without full inference.
- Pass1 header: `HEADER_OK`; keys 1024.
- Pass2 header: `HEADER_OK`; keys 1024.
- Base transformer header: `HEADER_OK`; keys 1024.
- Full GPU model load: not attempted in F0; no inference/training/optimizer step.
- Next gate: F1 official sample inference if sample data exists; otherwise F2 Gate8 inference under the documented sample-not-provided rule.

## Milestone F1 update - VOID_OFFICIAL_SAMPLE_INFERENCE_PASS

- Time: 2026-06-30T17:02:21+08:00
- Status: `VOID_OFFICIAL_SAMPLE_INFERENCE_PASS`
- Evidence: `reports/exp50_void_official_sample_inference.md`
- Official sample: `lime`; Pass1 only; raw frames 85.
- Output: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp50_pai_void_adapter_feasibility/f1_official_sample_lime/lime-fg=-1-0001.mp4`
- Runtime fix: used Exp50 runtime bundled-ffmpeg symlink because system `/usr/bin/ffmpeg` lacks `libblas.so.3`; no system or official-source modification.
- Visual review: Codex opened raw and tuple quick sheets; technical valid, not VOR quality evidence.
- Safety: no training, no optimizer step, no VOR-Eval, no hard comp, no VOID positive claim.
- Next gate: F2 VOR Gate8 inference smoke.
## Milestone F2 update - VOID_INFERENCE_SMOKE_PASS

- Time: 2026-06-30T17:24:15+08:00
- Status: `VOID_INFERENCE_SMOKE_PASS`
- Evidence: `reports/exp50_void_official_inference_smoke.md`, `reports/exp50_void_vor_gate8_metrics.csv`, and `reports/exp50_void_vor_gate8_visual_review.csv`
- Technical valid: 8 / 8
- Classification counts: {'MEDIUM_HARD_LOSER': 2, 'TOO_CLOSE': 2, 'VOID_OUTPUT_USABLE': 4}
- Usable or bounded loser: 6 / 8
- Systematic outside collapse: False
- Safety: no training, no optimizer step, no VOR-Eval, no hard comp, no VOID positive claim.
- Next gate: G0 adapter micro-data preparation only; optimizer steps remain locked behind G1.
## Milestone G0 update - VOID_ADAPTER_MICRO_DATA_READY

- Time: 2026-06-30T17:29:15+08:00
- Status: `VOID_ADAPTER_MICRO_DATA_READY`
- Evidence: `reports/exp50_void_adapter_micro_data.md`, `reports/exp50_void_adapter_micro_data.csv`, and `reports/exp50_void_adapter_micro_data_summary.json`
- Train/Heldout: 4 / 4
- Loser sources: {'controlled_local_corruption': 6, 'void_pass1_raw_medium_hard': 2}
- Scene overlap: False
- Source: VOR-Train only; VOR-Eval excluded: True
- Safety: no training, no optimizer step, no hard comp, no VOID positive claim.
- Next gate: G1 trainable forward / zero-gap / one-step only; direct 10-step remains locked.
## Milestone G1 update - VOID_TRAINABLE_FORWARD_BLOCKED

- Time: 2026-06-30T17:37:17+08:00
- Status: `VOID_TRAINABLE_FORWARD_BLOCKED`
- Exact blocker: `VOID_TRAINABLE_FORWARD_BLOCKED_PREFERENCE_WRAPPER_REQUIRED`
- Dataset view: official bucket `ImageVideoDataset` loads train4 successfully.
- Zero-gap: not run; official VOID script lacks required reference/winner-loser preference forward.
- One-step: not run; running SFT one-step would be off-protocol for this gate.
- Deepspeed: intentionally not installed; official 8-process shell path remains blocked pending controlled install.
- Safety: no training, no backward, no optimizer step, no 10-step, no VOID positive claim.
## Milestone H0 update - VOID_PREFERENCE_WRAPPER_REQUIRED_CONFIRMED

- Time: 2026-06-30T22:59:39+08:00
- Status: `VOID_PREFERENCE_WRAPPER_REQUIRED_CONFIRMED`
- Evidence: `reports/exp50_void_preference_wrapper_readback.md` and `reports/exp50_void_preference_wrapper_blocker.md`
- Finding: official VOID `train.py` is SFT MSE over scheduler noise/velocity targets, not a policy/reference winner-loser DPO forward.
- Single-process wrapper: allowed and preferred before any deepspeed install.
- Deepspeed: still intentionally uninstalled; only a controlled wheelhouse install is allowed if single-process proves impossible.
- VOID role: baseline / loser-generator candidate, not adapter evidence.
- Safety: no training, no optimizer step, no VOR-Eval, no hard comp, no VOID positive claim.
## Milestone H1 update - VOID_SFT_FORWARD_PARITY_EXPLAINED

- Time: 2026-06-30T23:08:07+08:00
- Status: `VOID_SFT_FORWARD_PARITY_EXPLAINED`
- Evidence: `reports/exp50_void_sft_forward_parity.md`, `reports/exp50_void_sft_forward_parity.csv`, `reports/exp50_void_sft_forward_parity_summary.json`
- Wrapper code: `exp50_pai_void_adapter_feasibility/void_preference_wrapper/void_sft_wrapper.py`
- Script: `exp50_pai_void_adapter_feasibility/scripts/run_void_sft_parity.py`
- Sample: `BLENDER_CON001_00636`; requested frames 17 -> effective frames 13 after official patch-size truncation.
- Target parameterization: `v_prediction`.
- SFT loss: 0.035282157361507416; finite: True.
- Missing/unexpected keys: 0 / 0.
- Safety: no training, no backward, no optimizer step, no VOR-Eval, no hard comp, no VOID positive claim.
## Milestone H2 update - VOID_PREFERENCE_FORWARD_PASS

- Time: 2026-06-30T23:15:48+08:00
- Status: `VOID_PREFERENCE_FORWARD_PASS`
- Evidence: `reports/exp50_void_preference_forward.md`, `reports/exp50_void_preference_forward.csv`, `reports/exp50_void_preference_forward_summary.json`
- Policy/reference: identical VOID pass1 clones; reference frozen.
- Trainable subset: `proj_out`; trainable params 393344 / 5571462784.
- Target parameterization: `v_prediction`.
- Same noise/timestep: True / True.
- Winner policy/reference loss: 0.0640571117401123 / 0.0640571117401123.
- Loser policy/reference loss: 0.08385218679904938 / 0.08385218679904938.
- DPO loss: 0.6931471824645996; preference margin: 0.0.
- Grad finite/reference grad zero: True / True.
- Safety: no optimizer step, no training loop, no VOR-Eval, no hard comp, no VOID positive claim.
## Milestone H3 update - VOID_ZERO_GAP_PASS

- Time: 2026-06-30T23:17:43+08:00
- Status: `VOID_ZERO_GAP_PASS`
- Evidence: `reports/exp50_void_zero_gap_gate_v2.md`, `reports/exp50_void_zero_gap_gate_v2.csv`, `reports/exp50_void_zero_gap_gate_v2_summary.json`
- Winner/loser gaps: 0.0 / 0.0.
- DPO loss: 0.6931471824645996 vs log(2) 0.6931471805599453.
- Reference grad zero / policy grad finite: True / True.
- Safety: no optimizer step, no training loop, no VOR-Eval, no hard comp, no VOID positive claim.
## Milestone H4 update - VOID_ONE_STEP_PARETO_MIXED

- Time: 2026-06-30T23:24:57+08:00
- Status: `VOID_ONE_STEP_PARETO_MIXED`
- Evidence: `reports/exp50_void_one_step_gate_v2.md`, `reports/exp50_void_one_step_diagnostics_v2.csv`, `reports/exp50_void_one_step_visual_review_v2.csv`, `reports/exp50_void_one_step_summary_v2.json`
- Optimizer: AdamW lr=1e-05 weight_decay=0.0 grad_clip=1.0.
- Optimizer steps: 1.
- Loss before step: 0.6931471824645996.
- Grad finite / norm: True / 0.011962890625.
- Param delta positive / max norm: True / 0.005055009387433529.
- Reload ok: True.
- Heldout forward finite: True.
- Step1 vs Step0 L1: 0.019980037584900856.
- Video inference generated: False.
- Gate note: H5 10-step remains locked because H4 is not `VOID_ONE_STEP_PASS`.
- Safety: exactly one optimizer step; no long training, no VOR-Eval, no hard comp, no VOID positive claim.

## Exp50 VOID Preference-Wrapper Outcome - 2026-06-30T23:30:58+08:00

Status: `VOID_TRUE_ADAPTER_FEASIBLE_NEEDS_MICRO_TRAINING`.

VOID official inference is usable on VOR-Train Gate8 (`VOID_INFERENCE_SMOKE_PASS`, technical valid 8/8). The isolated VOID-native wrapper reached `VOID_SFT_FORWARD_PARITY_EXPLAINED`, `VOID_PREFERENCE_FORWARD_PASS`, and `VOID_ZERO_GAP_PASS` without modifying official VOID source, shared trainer, or `inference/metrics.py`.

H4 one-step is `VOID_ONE_STEP_PARETO_MIXED`, not PASS. Exact blocker: `VOID_ONE_STEP_VIDEO_HELDOUT_EVIDENCE_MISSING`. H5 10-step was not run. VOID remains not third adapter evidence. Next minimal experiment: H4b one-step video heldout evidence using the saved step1 adapter; only then consider 10-step.

## Exp50 H4b-0 One-Step Evidence Readback - 2026-07-01T00:05:40+08:00

Status: `VOID_ONE_STEP_EVIDENCE_READBACK_DONE`. One-step adapter checkpoint `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp50_pai_void_adapter_feasibility/one_step_v2/adapter_proj_out_step1.pt` and train4/heldout4 manifests are identified. Target parameterization remains `v_prediction` and trainable subset `proj_out`. 10-step remains locked until heldout video evidence upgrades one-step to PASS.

## Exp50 H4b-1 One-Step Checkpoint Audit - 2026-07-01T00:08:01+08:00

Status: `VOID_ONE_STEP_CHECKPOINT_READY`. Adapter checkpoint `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp50_pai_void_adapter_feasibility/one_step_v2/adapter_proj_out_step1.pt` exists, SHA256 `849326121699e51673990df7bef52c578245812ad934a88bbe6f2acf93b972d5`, and adapter keys match `proj_out.weight` / `proj_out.bias`. H4 strict reload was already OK. H4b-2 video generation waits for a free PAI GPU; no unrelated GPU process was killed.

## Exp50 H4b-2 One-Step Heldout Generation - 2026-07-01T00:09:39+08:00

Status: `VOID_ONE_STEP_HELDOUT_GENERATION_BLOCKED`. Heldout4 Step0/Step1 video generation was not launched because all 8 PAI GPUs were occupied by unrelated root jobs. No process was killed. H4b-3 and H5 remain locked.
