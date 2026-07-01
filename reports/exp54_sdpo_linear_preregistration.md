# Exp54 SDPO / Linear-DPO Preregistration

Status: `EXP54_SDPO_LINEAR_PREREGISTERED`

Requested output root `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp54_void_sdpo_linear_pai` is not writable by `hj`, so large artifacts will use `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp54_void_sdpo_linear_pai/outputs`. This is recorded as `OUTPUT_ROOT_PERMISSION_FALLBACK_RUNTIME`; no files are written into the blocked requested output root.

Cache root: `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp54_void_sdpo_linear_pai/outputs/cache/tensor_cache`  
Cache files: `32`

## Wave PAI-1

- GPU0: `R3_Q2_T500_S0`
- GPU1: `R4_Q2_T500_S0`

## Conditional Waves

- PAI-2: `R3_Q1_T500_S0`, `R4_Q1_T500_S0` only if Wave1 has no PASS but at least one mixed-safe cell.
- PAI-3: `R3_Q2_T300_S0`, `R4_Q2_T300_S0` only if T500 is promising but affected/overlap still regresses.

No VOR-Eval, hard comp, direct 10-step, or third-backbone claim is allowed.
