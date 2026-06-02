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

## Stage-Aware Correction

DiffuEraser Stage1 and Stage2 should be interpreted separately:

| Stage | Main role |
| --- | --- |
| Stage1 | spatial generation quality, BrushNet, UNet2D, appearance |
| Stage2 | video temporal consistency, motion module, temporal modeling |

The Exp7-PM-Gate1500 result means:

- DPO Stage1 learned useful partial-mask spatial behavior.
- DPO Stage2 damaged the candidate under the current loser-driven objective.
- The next candidate is **DPO Stage1 + frozen SFT Stage2**, not Stage1-only
  inference.

The desired hybrid is:

```text
spatial / appearance weights = Exp7 DPO Stage1
temporal / motion weights = validated SFT Stage2
```

Do not simply overwrite the DPO checkpoint with a full SFT Stage2 checkpoint,
because full Stage2 exports may contain spatial and temporal weights. The
hybrid builder must keep DPO spatial modules and preserve only SFT temporal /
motion modules.

Prepared next audit:

```text
script = scripts/eval_exp7_dpoS1_sftS2_hybrid_partialmask.sh
builder = tools/build_diffueraser_dpoS1_sftS2_hybrid.py
inspector = tools/inspect_diffueraser_stage_weights.py
output = logs/partialmask_eval/exp7_pm_dpoS1_sftS2_hybrid_<timestamp>/
final_report = reports/exp7_dpoS1_sftS2_hybrid_eval_report.md
```

No new long training, DPO Stage2, full Exp7, full VBench, Exp8, D2 regen, or
D3 full run should be started before this hybrid audit is reviewed.
## Mainline Correction: VideoDPO Bridge vs Target Domain

After visually inspecting generated videos, the current interpretation is:

- DiffuEraser-base remains visually stronger than Exp7 DPO variants on sampled
  partial-mask videos.
- Exp7 DPO Stage1 shows flicker and high-frequency artifacts in the masked
  region.
- Exp7 DPO Stage1 + DPO Stage2 can form more stable structures in some samples
  but still has temporal artifacts and wrong-object priors.
- The official/base Stage2 hybrid did not rescue the Exp7 DPO Stage1 output.

This does not motivate jumping directly to Exp8. The larger issue is that
VideoDPO is a bridge domain, while final evaluation is YouTube-VOS / DAVIS.
Therefore the next decision gate is target-domain evaluation of existing
checkpoints, not more VideoDPO loss tuning.

Do not start VideoDPO SFT warmup. Do not start Exp8. Do not start Exp9 training
until target-domain eval determines whether VideoDPO-bridge DPO transfers.

## D3 H20 Audit Result

H20 D3 root:

```text
/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4
```

Audit report:

```text
/home/nvme01/H20_Video_inpainting_DPO/reports/d3_h20_audit_report.md
```

Key numbers:

- size: 249G
- file count: 1,819,879
- shard count: 3,327
- candidates: 13,308
- selected primary comp: 3,327
- selected primary no-comp: 3,327
- sampled selected-primary-comp rows: 100/100 status OK
- sampled win/mask/final-loser frame checks: 300/300 readable
- frame count/resolution: 16 frames, 512x320
- path issue: all sampled manifest paths are H20-only `/home/nvme01/...`

Conclusion: D3 is a valid-looking target-domain generated-loser asset on H20,
but PAI must use repaired/path-rewritten manifests after sync.
