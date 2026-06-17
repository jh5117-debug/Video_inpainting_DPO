# Qualitative Summary

Fixed DAVIS50 two-row OR visual grids were generated on PAI and synced back to HAL.

PAI fixed:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp15_or_benchmark_davis50_fixed/visual_grids
```

HAL fixed:

```text
/home/hj/dpo-2-1-exp/exp15_or_benchmark_davis50_visuals_fixed
```

The previous HAL directory is retained only as bug evidence because it used less
portable OpenCV/mp4v encoding. Fixed MP4s are H.264 / yuv420p / libx264.

The visual grid layout includes input and mask overlay plus requested method
slots. Blocked methods are rendered as explicit unavailable placeholders.

The 20-case list is metric-preselected and stored at:

```text
reports/exp15_or_davis50_paper_ready_cases.md
reports/exp15_or_davis50_paper_ready_cases.csv
```

Important caveat: OR has no GT for the removed foreground region, so final paper
figure selection still needs human visual review of mask-inside removal quality.
