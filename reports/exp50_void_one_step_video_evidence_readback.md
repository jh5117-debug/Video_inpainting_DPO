# Exp50 VOID One-Step Video Evidence Readback

Time: 2026-07-01T00:05:40+08:00

Status: `VOID_ONE_STEP_EVIDENCE_READBACK_DONE`

## Confirmed Inputs

1. One-step adapter checkpoint: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp50_pai_void_adapter_feasibility/one_step_v2/adapter_proj_out_step1.pt`
2. Step0 policy/reference: original VOID pass1 transformer loaded from `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/void/netflix_void-model/void_pass1.safetensors` with base config `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/void/CogVideoX-Fun-V1.5-5b-InP`.
3. Train4 manifest: `manifests/exp50_void_adapter_train4.jsonl`
4. Heldout4 manifest: `manifests/exp50_void_adapter_heldout4.jsonl`
5. Output root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp50_pai_void_adapter_feasibility/one_step_heldout_evidence_v2`
6. Wrapper paths: `exp50_pai_void_adapter_feasibility/void_preference_wrapper/void_sft_wrapper.py`, `exp50_pai_void_adapter_feasibility/scripts/run_void_one_step.py`, and H4b evidence scripts to be added under `exp50_pai_void_adapter_feasibility/scripts/`.
7. Target parameterization: `v_prediction`.
8. Trainable subset: `proj_out`.
9. Policy/reference identity: H2/H3 used identical pass1 clones; H4 Step1 saved only adapter subset after exactly one AdamW step.
10. VOR-Eval: not used.
11. 10-step lock: still locked until H4b upgrades one-step to `VOID_ONE_STEP_PASS`.

## Current One-Step State

- H4 status: `VOID_ONE_STEP_PARETO_MIXED`.
- Optimizer: AdamW lr=1e-05 weight_decay=0.0 grad_clip=1.0.
- Param delta positive: True; max delta norm 0.005055009387433529.
- Strict reload OK: True.
- Heldout forward finite: True.
- Video inference generated: False.
- Exact blocker: `VOID_ONE_STEP_VIDEO_HELDOUT_EVIDENCE_MISSING`.

## Safety

This readback did not train, did not run optimizer steps, did not run inference, did not use VOR-Eval, and did not use hard comp.
