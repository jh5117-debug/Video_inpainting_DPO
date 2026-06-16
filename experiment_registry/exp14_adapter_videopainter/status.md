# Status

Status: completed_negative_gate.

PAI sync strategy: clean_worktree.

Clean repo:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate
```

What passed:

- Exp14 isolated trainer and launcher are present.
- Static checks passed.
- VideoPainter code repo is present.
- VideoPainter / CogVideoX weights are present and validated.
- YouTube-VOS, DAVIS, and generated-loser manifest are present.
- Trainer preflight passed.
- Gate2000 completed 2000 steps.
- `checkpoint-500`, `checkpoint-1000`, `checkpoint-1500`, `checkpoint-2000`, and `last_weights` exist.
- DAVIS50 eval completed with the Exp14 thin eval adapter.

DPO diagnostics:

```text
DPO_SATURATED
LOSER_DOMINANT
GRAD_SPIKE_OBSERVED
```

DAVIS50 result:

| method | PSNR | SSIM | strict mask PSNR |
|---|---:|---:|---:|
| VideoPainter baseline | 31.6124 | 0.9608 | 19.9691 |
| VideoPainter + DPO adapter | 29.8028 | 0.9580 | 18.1595 |

Decision:

This is a completed negative adapter gate. The current Exp11-style direct DPO
branch adapter should not be continued as a longer VideoPainter run without a
new adapter design.
