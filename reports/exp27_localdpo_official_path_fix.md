# Exp27 LocalDPO Official Path Fix

Status: `FAITHFUL_LOCALDPO_OFFICIAL_MASK_DIGEST_PASSED`

The previous LocalDPO smoke reported `blocked_official_code_missing` because the
adapter looked under `/home/hj/video_dpo_paper_code_cache/repos`. On PAI the
pinned official cache is located at:

`/mnt/nas/hj/video_dpo_paper_code_cache/Local-DPO_7528e966b17283cfa638577827e456737335f030/innerT2V/utils/random_mask_gen.py`

The Exp27 official parity helper now resolves both plain `Local-DPO/` and
commit-suffixed `Local-DPO_*` cache layouts without modifying the official
clone.

CPU verification:

- official mask digest: passed
- shape: `[13, 48, 80]`
- sum: `1103895`
- mean: `22.11328125`
- sha256: `d1e7686f09e942bb4d3c24b38e19e187ef5108f66cacdd77bdce6c4773cc09fd`
- matplotlib RGB shim: installed
- one-step original-loss smoke: passed
- ten-step original-loss smoke: passed
- RC-FPO: not started

Output JSON:

`reports/exp27_localdpo_official_path_fixed_smoke.json`

PAI output:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_26_27_followup_20260624/exp27_localdpo_six_video_smoke_official_path_fixed`
