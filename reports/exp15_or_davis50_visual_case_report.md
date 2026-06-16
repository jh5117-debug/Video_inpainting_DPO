# Exp15 DAVIS50 OR Visual Case Report

- Visual grid root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp15_or_benchmark_davis50/visual_grids`
- Manifest: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp15_or_benchmark_davis50/visual_grids/visual_manifest.csv`
- HAL copy: `/home/hj/dpo-2-1-exp/exp15_or_benchmark_davis50_visuals`
- Rows: two-row grid with input/mask plus available methods; blocked methods are explicit placeholders.
- This report is generated after DAVIS50 inference and does not use composited frames.

## Completed Outputs

- videos: 50 DAVIS50 grid MP4s
- contact sheets: 50 DAVIS50 grid JPGs

## Method Slots

1. ProPainter
2. VideoComposer / VideoComp
3. CoCoCo
4. FloED
5. DiffuEraser SFT-48000
6. VideoPainter
7. VACE
8. Ours Exp11 outer b0.75 S2

Blocked methods are shown as `BLOCKED / N.A.` placeholders, not silently
removed.

## Candidate Cases

See:

```text
reports/exp15_or_davis50_paper_ready_cases.md
reports/exp15_or_davis50_paper_ready_cases.csv
```

The 20-case list is metric-preselected. Because OR has no mask-inside GT, final
paper figure selection still needs human visual review of the removal region.
