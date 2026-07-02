# Exp60B Status

Status:

- `EXP60B_READBACK_DONE`
- `EXP60B_VPDATA_AVAILABLE`
- `EXP60B_PAI_GPU_READY`
- `EXP60B_H20_READY_VIA_PAI_RELAY`
- `EXP60B_VPDATA_SUBSET_PLAN_READY`
- `EXP60B_H20_VPDATA_SUBSET_BLOCKED_NETWORK`
- `EXP60B_H20_HF_MIRROR_READY`

Current continuation: H20 can reach `https://hf-mirror.com`; direct
`huggingface.co` remains blocked. Next step is a guarded HF-mirror download
attempt, then clash fallback only if needed. Full VPData clone/download remains
forbidden.
