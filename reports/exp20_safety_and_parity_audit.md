# Exp20 Safety And Parity Audit

## Worktrees

| Host | Path | Branch | HEAD |
| --- | --- | --- | --- |
| HAL | `/home/hj/H20_Video_inpainting_DPO_exp20_autoresearch` | `research/exp20-adaptive-region-autoresearch-20260619` | `b020ace2ebdee9565a35761946b033bf2b8c3aa8` |
| PAI | `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp20_autoresearch` | `research/exp20-adaptive-region-autoresearch-20260619` | `b020ace2ebdee9565a35761946b033bf2b8c3aa8` |

Safety audits:

- HAL: `reports/exp20_hal_safety_audit.txt`
- PAI: `reports/exp20_pai_safety_audit.txt` in the PAI worktree

## Source Of Truth

Current best remains `Exp11 boundary outer b0.75 S2`.

Canonical protocol:

```text
raw6, hard comp, D+G off, no PCM, no mask dilation, no Gaussian blur,
frame-wise in-memory metrics via inference/metrics.py
```

## Implementation Status

Implemented:

- legacy latent exact boundary maps;
- image-space Euclidean outer boundary maps;
- adaptive area/perimeter and sqrt-area radius calculation;
- legacy global weighted DPO aggregation;
- region-balanced DPO aggregation;
- safe config-only search controller.

Pending before PAI search:

- full trainer parity against Exp11 on same batch/noise/timestep;
- locked dev split and overlap audit;
- SFT / Exp11 dev baseline recomputation.

Status:

```text
PRECHECK_IMPLEMENTED_NOT_TRAINING_READY
```
