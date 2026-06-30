# Exp50 VOID Next Steps

Time: 2026-06-30T23:30:58+08:00

Recommended next minimal experiment: `Run a narrowly scoped H4b one-step video heldout evidence gate using the saved step1 adapter; run 10-step only if H4b upgrades one-step to PASS.`

## Immediate Next Step

Run H4b only:

- Load the saved one-step adapter checkpoint: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp50_pai_void_adapter_feasibility/one_step_v2/adapter_proj_out_step1.pt`.
- Generate heldout4 Step0 and Step1 video outputs, or at minimum one heldout video if memory/runtime requires a narrowed smoke.
- Compute PSNR, SSIM, mask/object PSNR, affected PSNR, boundary PSNR, outside PSNR/L1, and temporal flicker.
- Open and review all generated evidence sheets.
- Upgrade H4 to `VOID_ONE_STEP_PASS` only if Step1 is finite, noncollapsed, visually safe, and not outside-damaging.

## Locked Until H4b Passes

- Do not run H5 10-step.
- Do not run 30/50/100/300/500 steps.
- Do not run DDP/deepspeed.
- Do not use VOR-Eval.

## If H4b Passes

Run H5 10-step micro gate only on train4/heldout4, then evaluate heldout4 visually and metrically. Keep the claim at micro-gate level unless heldout metrics and visual review are clearly positive.

## If H4b Fails

Treat VOID as inference baseline / loser generator only, and either return to ROSE feasibility or pause the third-model search.
