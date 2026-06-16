# Exp15 OR Benchmark

This experiment is for object-removal evaluation, not adapter training.

Scope:

- DAVIS50 object-removal subset using real DAVIS2017 foreground masks.
- YouTubeVOS100 object-removal subset using the existing fixed-seed eval100 set.
- Baseline comparison is frozen inference only.
- No DPO training and no adapter training.

Target methods:

```text
MiniMax-Remover
CoCoCo
FloED
DiffuEraser SFT-48000
VideoPainter
VACE
VideoComp / VideoComposer if runnable
DiffuEraser Exp11 outer b0.75 S2
```

Large outputs, weights, videos, and datasets must stay outside git.
