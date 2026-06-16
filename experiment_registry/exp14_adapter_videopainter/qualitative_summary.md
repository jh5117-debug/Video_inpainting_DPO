# Qualitative Summary

Status: completed_four_column_visuals.

Four-column output path on PAI:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/logs/target_eval/exp14_videopainter_adapter_gate2000_davis
```

Columns:

1. GT
2. mask overlay
3. VideoPainter baseline
4. VideoPainter + DPO adapter

Generated artifacts:

- 50 side-by-side mp4 files;
- 50 contact sheets;
- frame-by-frame image folders;
- per-video metric table.

Positive candidates:

```text
rollerblade
scooter-black
dog-agility
bus
motorbike
libby
bear
flamingo
```

Failure candidates:

```text
hockey
paragliding-launch
hike
car-turn
dog
dance-jump
bmx-bumps
swing
```

Interpretation: visual review should focus on these cases, but metrics already
show the adapter is not a reliable overall improvement. It should not be used as
a paper-positive adapter result.
