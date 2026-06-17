# Status

Current status: DAVIS50-only OR benchmark completed for runnable methods.

Completed methods:

- ProPainter
- DiffuEraser SFT-48000
- Ours Exp11 outer b0.75 S2

Blocked methods stayed in tables and grids with explicit labels:

- VideoComposer / VideoComp
- CoCoCo
- FloED
- VideoPainter
- VACE
- MiniMax-Remover

Outputs:

```text
reports/exp15_or_davis50_quantitative_summary.csv
reports/exp15_or_davis50_visual_case_report.md
reports/exp15_or_davis50_paper_ready_cases.md
/home/hj/dpo-2-1-exp/exp15_or_benchmark_davis50_visuals
```

2026-06-17 audit update:

- old OpenCV/mp4v visual grids are retained only as bug evidence;
- fixed H.264/yuv420p visual grids completed at `/home/hj/dpo-2-1-exp/exp15_or_benchmark_davis50_visuals_fixed`;
- fixed metrics use `TC_bg_pixel_proxy` and are not directly comparable to MiniMax paper TC;
- DAVIS90 future manifest prepared but not run.
