# Exp60B Status

Status:

- `EXP60B_READBACK_DONE`
- `EXP60B_VPDATA_AVAILABLE`
- `EXP60B_PAI_GPU_READY`
- `EXP60B_H20_READY_VIA_PAI_RELAY`
- `EXP60B_VPDATA_SUBSET_PLAN_READY`
- `EXP60B_H20_VPDATA_SUBSET_BLOCKED_NETWORK`

Blocked next step: H20/PAI outbound network cannot reach Hugging Face for the
official VPData metadata/raw download path. Enable H20 egress, authorize a
HAL-first download route, or provide an internal mirror. Full VPData
clone/download remains forbidden.
