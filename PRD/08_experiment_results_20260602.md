# Experiment Results Through 2026-06-02

This file is the short current experiment ledger. Detailed setup remains in
`00_current_status.md`, `01_experiment_matrix.md`, and
`07_exp5_exp6_winner_anchored_rerun.md`.

## Completed / Diagnosed

| Experiment | Status | Key conclusion |
| --- | --- | --- |
| Exp3 VideoDPO -> DiffuEraser bridge | success reference | Shows the base DiffuEraser DPO bridge can run; later Exp5 failures are not task plumbing failures. |
| Old Exp5 beta500, 10000 + 10000 | failed / collapsed / diagnostic only | D2 comp + full-mask full-loss + beta500 + no SFT reg + long training over-optimized DPO and produced high-frequency noise / color explosion. |
| Exp5 beta10, 4000 + 4000 unanchored | failed / collapsed / diagnostic only | Stage2 loaded Stage1 correctly, but the unanchored DPO objective still found a degenerate ranking solution. |
| Exp5 winner-anchored comp | improved but not final | Winner-gap regularization reduced universal stripe collapse, but full-mask/full-video data-only outputs still show texture/color attractors. |
| Exp7 gate1500 full-mask qual30 | failed / task-mismatched | This eval is not a fair final Exp7 verdict because Exp7 trains partial-mask inpainting. |
| Exp7-PM-Gate1500 partial-mask eval | Stage1 metric pass; Stage2 regression | Task-matched partial-mask eval shows Stage1_last beats DiffuEraser-base, but Stage2_last regresses. |

## Active / Pending

| Experiment | Status | Next action |
| --- | --- | --- |
| Exp6 winner-anchored no-comp | running on H20 Stage2 | Continue; monitor only. It uses GPU 0-5 and leaves GPU 6/7 idle. |
| Exp7 no-lose-gap gate1000 | script prepared only | Do not launch automatically. Use if Exp7 visual review confirms loser-degradation artifacts. |
| Exp8 region loss | not started | Do not change in the current round. |

## Exp7-PM-Gate1500 Metrics

Output:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/partialmask_eval/exp7_gate1500_20260602_000500
```

| Model | mask-region PSNR | mask-region SSIM | Interpretation |
| --- | ---: | ---: | --- |
| DiffuEraser-base | 8.99765 | 0.272146 | baseline |
| Exp7 Stage1_last | 9.57079 | 0.288404 | best evaluated checkpoint; beats base |
| Exp7 Stage2_last | 7.88448 | 0.235938 | regressed |

Decision:

- Exp7 partial-mask task alignment is promising.
- Do not launch full Exp7 4000+4000 yet.
- Review the 60 partial-mask side-by-side videos.
- Prefer a Stage1-focused or no-lose-gap follow-up over another long Stage2
  with the same loser-degradation incentive.

## Code Notes

The partial-mask eval tool now decodes input videos with ffmpeg rawvideo rather
than `imageio.get_reader`. PAI selected an incompatible `pyav` backend and
failed on `ContainerFormat.variable_fps`; the ffmpeg reader was smoke-tested on
generated Exp7 eval mp4 files.
