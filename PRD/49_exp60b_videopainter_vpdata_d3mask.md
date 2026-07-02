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

Status: `EXP60B_READBACK_DONE`

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
- H20 SSH alias was not resolvable from HAL, so H20 download is blocked until
  the real H20 host/route is provided or recovered.

## Required Next Gate

Before any data download:

1. Recover H20 SSH route or explicitly authorize PAI/NAS-first subset download.
2. Implement a file-level and row-level subset downloader that downloads only
   train1000/test100 selected rows.
3. Do not run `git clone https://huggingface.co/datasets/TencentARC/VPData` as
   a full data checkout.
4. Do not run official `VPData_download.py` unmodified, because it iterates all
   Pexels rows.

