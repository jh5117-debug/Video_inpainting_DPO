# Exp50 VOID One-Step Gate V2

Status: `VOID_ONE_STEP_PARETO_MIXED`

## Optimizer

- Optimizer: AdamW
- LR: 1e-05
- Weight decay: 0
- Grad clip: 1.0
- Trainable filter: `proj_out`
- Optimizer steps: 1

## Checks

- Loss before step: 0.6931471824645996
- Grad finite: True
- Max adapter param delta norm: 0.005055009387433529
- Adapter checkpoint: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp50_pai_void_adapter_feasibility/one_step_v2/adapter_proj_out_step1.pt`
- Reload ok: True; missing=[]; unexpected=[]
- Heldout forward finite: True
- Step1 vs Step0 prediction L1: 0.019980037584900856
- Video inference generated: no

## Interpretation

Technical one-step checks passed, but this is conservatively marked `VOID_ONE_STEP_PARETO_MIXED` rather than `VOID_ONE_STEP_PASS` because no video-level heldout inference/visual evidence was generated in this gate. Therefore H5 10-step remains locked.

## Safety

Exactly one optimizer step was run. No VOR-Eval, hard comp, deepspeed install, long training, or VOID positive claim was made.
