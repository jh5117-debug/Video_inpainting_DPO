# Exp25 Overnight GPU2 Gate32 Completion

Date: 2026-06-23 UTC

Runtime root:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260623`

## Controller

- Original controller PID `1903925` did not claim newly free GPU2 because it predated the GPU2 continuation tasks.
- Updated controller commit: `22f7c54f49c46655700d7cb4193ac18dfa3bf037`.
- Fix: zombie child processes are now treated as completed instead of alive.
- Active controller after fix: PID `2041604`.
- Controller snapshot:
  `/mnt/workspace/hj/nas_hj/runtime_code_snapshots/exp25_22f7c54f_zfix`

## Completed Queue

All requested GPU2 tasks completed in order:

1. Exp26 VideoPainter official 49F Probe4 inference.
2. Exp27 SDPO real-batch parity.
3. Exp27 Linear-Frozen / Linear-EMA real-batch parity.
4. Exp25 canonical Gate32 yield calibration.

Final queue status:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260623/overnight_queue.csv`

## Gate32 Result

Gate32 materialization:

- Manifest:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/materialized/gate32_canonical_d0_24f/gate32_materialized.jsonl`
- Result: `32/32` materialized.

Gate32 DiffuEraser canonical OR generation:

- Output root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/preference_candidates/gate32_canonical_raw6_d0`
- Summary:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/preference_candidates/gate32_canonical_raw6_d0/diffueraser_smoke_summary.json`
- Result: `32/32` OK.
- Frames: `24` for every sample.
- Generator ID: `diffueraser_or_none_propainter_abd3ad48f60f`.
- `pcm_mode`: `none`.
- `prior_mode`: `propainter`.
- `hard_comp`: `false`.

This is yield calibration only. It is not a long training run, and VOR-Eval was not used.

## GPU State

After Gate32 completion:

- GPU2: `0 MiB / 143771 MiB`, utilization `0%`.
- GPU7: stale allocation remains about `58071 MiB`; it was not used.

Manual monitor log:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260623/manual_monitor_5min.log`

