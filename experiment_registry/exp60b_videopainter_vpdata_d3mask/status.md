# Exp60B Status

Status:

- `EXP60B_READBACK_DONE`
- `EXP60B_VPDATA_AVAILABLE`
- `EXP60B_PAI_GPU_READY`
- `EXP60B_H20_READY_VIA_PAI_RELAY`
- `EXP60B_VPDATA_SUBSET_PLAN_READY`
- `EXP60B_H20_VPDATA_SUBSET_BLOCKED_NETWORK`
- `EXP60B_H20_HF_MIRROR_READY`
- `EXP60B_H20_PEXELS_RAW_PROXY_REQUIRED`
- `EXP60B_H20_VPDATA_SUBSET_BLOCKED_PROXY`
- `EXP60B_HAL_VPDATA_SUBSET_BLOCKED`

Current continuation: H20 hf-mirror download completed partially: 1,089/1,100
raw videos downloaded, 11 Pexels raw URLs failed. H20 clash proxy fallback was
then run with resume and remained at 1,089/1,100, with the same 11 Pexels raw
URLs blocked by HTTP 403. HAL fallback probed those 11 missing locked URLs and
also received HTTP 403 for all 11. The exact train1000/test100 subset is still
blocked; PAI ready transfer is not allowed yet. Full VPData clone/download
remains forbidden.
