# Exp50 VOID Next Steps V2

Time: 2026-07-01T00:11:20+08:00

Next minimal action: resume H4b-2 when one PAI GPU is genuinely free.

## Resume Procedure

1. Re-check `nvidia-smi` and `nvidia-smi pmon`.
2. If at least one GPU is free, run heldout4 Step0/Step1 video generation using the existing adapter checkpoint `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp50_pai_void_adapter_feasibility/one_step_v2/adapter_proj_out_step1.pt`.
3. Compute heldout4 Step1 - Step0 metrics and inspect all evidence sheets.
4. Upgrade one-step only if the H4b decision rule passes.
5. Run 10-step only after `VOID_ONE_STEP_PASS`.

## Do Not Do Yet

- Do not run 10-step while H4b is blocked.
- Do not run 30/50/100/300/500 steps.
- Do not install deepspeed.
- Do not use VOR-Eval.
- Do not use hard comp.

## If GPU Remains Blocked

Keep VOID positioned as baseline / loser generator / adapter engineering candidate, not as third adapter evidence.
