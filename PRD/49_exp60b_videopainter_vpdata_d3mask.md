# PRD 49: Exp60B VideoPainter VPData D3-Mask PAI DPO Scaleup

Date: 2026-07-02

## Objective

Exp60B moves VideoPainter evidence away from the old VOR-BG primary32 setting
and toward a VPData-subset diagnostic with DiffuEraser-D3-equivalent masks.

The experiment is explicitly scoped as a subset scale-up:

- VPData train subset: 1000 videos.
- VPData test subset: 100 videos.
- Main mask policy: DiffuEraser D3 / partial-mask K4 equivalent.
- Winner: clean VPData video.
- Loser: VideoPainter official-base self-loser under the same source and mask.
- Training gates: 50 steps first; 100 steps only if 50-step is safe.
- No 2000-step in this experiment.

## Claim Boundary

This experiment must not be written as full VPData validation, final SOTA, or a
universal adapter claim. It is intended to test whether the old VOR-BG concern
is reduced by a larger VPData subset with D3-style masks.

## Current Status

Status: `EXP60B_READBACK_DONE` / `EXP60B_H20_READY_VIA_PAI_RELAY` /
`EXP60B_VPDATA_SUBSET_PLAN_READY` /
`EXP60B_H20_VPDATA_SUBSET_BLOCKED_NETWORK` /
`EXP60B_H20_PEXELS_RAW_PROXY_REQUIRED` /
`EXP60B_H20_VPDATA_SUBSET_BLOCKED_PROXY` /
`EXP60B_HAL_VPDATA_SUBSET_BLOCKED` /
`EXP60B_TRANSFER_BLOCKED`

Milestone A completed from the HAL Codex session:

- VPData official source is public on Hugging Face as `TencentARC/VPData`.
- The dataset page reports 392,077 rows and about 1.87 TB total size.
- Official docs state VPData contains mask/text annotations; VideoVo raw videos
  are uploaded in VPData, while Pexels raw videos are downloaded by URL from
  `pexels.csv` using `data_utils/VPData_download.py`.
- Full `git clone` or the unmodified Pexels download script would violate the
  Exp60B boundary because they target full VPData.
- PAI host `hj@47.103.26.60` is reachable and GPU0/GPU1 were idle during the
  readback probe.
- H20 SSH alias was not resolvable from HAL, but PAI relay to H20 is available:
  `hj@47.103.26.60` -> `ubuntu@27.190.15.128`.
- H20 `/home/nvme01` has 1.2T free, so storage passes the Exp60B hard stop.
- A guarded Pexels-only train1000/test100 download plan has been generated.
- H20 official metadata download is blocked by outbound network:
  `urllib.error.URLError: [Errno 101] Network is unreachable`.
- PAI also cannot reach Hugging Face from the urllib probe.
- Continuation result: H20 hf-mirror unblocked metadata and downloaded
  1,089/1,100 Pexels raw videos. The remaining 11 failed at source URL level
  and required clash proxy fallback.
- H20 clash proxy fallback remained at 1,089/1,100 with the same 11 HTTP 403
  Pexels raw URL failures.
- HAL fallback probed the 11 missing URLs directly and also received HTTP 403
  for all 11.
- The exact train1000/test100 subset is incomplete; PAI ready transfer and PAI
  manifests are blocked.

## Required Next Gate

Before any D3 mask generation:

1. Resolve the 11 locked Pexels raw URL failures, or open a new preregistered
   replan to replace exactly those blocked rows.
2. Verify 1,100/1,100 files on PAI/NAS with sha256 and decode checks.
3. Do not run `git clone https://huggingface.co/datasets/TencentARC/VPData` as
   a full data checkout.
4. Do not run official `VPData_download.py` unmodified, because it iterates all
   Pexels rows.
5. Do not generate D3 masks, losers, or DPO data from the partial 1,089-row set.
