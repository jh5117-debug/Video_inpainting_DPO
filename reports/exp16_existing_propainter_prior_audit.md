# Exp16 Existing ProPainter Prior Audit

Date: 2026-06-17

## Manifests Checked On PAI

Primary current-best training manifest:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/exp09_10_11_youtubevos_gtwin_d3comp_pai/manifests/selected_primary_comp.gtwin.pai_paths.jsonl
```

Earlier official generated-loser manifest:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4/manifests/selected_primary_comp.repaired.pai_paths.jsonl
```

Both manifests contain:

- `win_video_path`
- `final_loser_video_path`
- `raw_loser_video_path`
- `comp_loser_video_path`
- `mask_path`
- `diffueraser_prior_mode = unconfirmed`

Neither manifest contains verified fields such as:

- `prior_frame_dir`
- `propainter_prior_frame_dir`
- `propainter_frame_dir`
- `prior_video_path`
- `propainter_video_path`

## Existing ProPainter Directory Search

The generated-loser tree contains older archived `propainter` folders under
abandoned D2/VideoDPO test archives. Those paths are not aligned with the
current Exp9/10/11 GT-win YouTube-VOS training manifest and must not be reused
as Exp16 training priors.

## Conclusion

```text
current_prior_status = missing
```

Exp16 must generate a new real ProPainter prior cache for the current GT-win
manifest. The next safe step is `limit=100` cache generation, not training.
