# Exp25 DiffuEraser OR Root-Cause Matrix

Status: `DIFFUSERASER_OR_PROTOCOL_ROOT_CAUSE_PENDING`

This report is intentionally not marked complete. The current run only finished
the individual Gate32 frame/crop reaudit. The requested DE-A through DE-F stack
matrix has not yet been run, so no scientific conclusion is recorded about
whether the yield problem comes from the model, checkpoint, scheduler, mask
dilation, raw/no-comp protocol, ProPainter prior, or OR preprocessing.

Current evidence from the individual reaudit:

- Whole-video black-frame collapse: not observed (`black_frame_ratio=0.0` on all 32 rows).
- Contact-sheet purple/black appearance: partly explained by error-map visualization columns, not necessarily raw output.
- Raw loser quality: still poor in 21/32 rows, dominated by large task-region mismatch.
- Outside/background damage: not the dominant failure; many trivial-bad samples retain reasonable outside PSNR.
- Hard comp: not used; `hard_comp=false`, `comp_mode=none`.

Pending stack matrix:

| stack | intended comparison | status |
| --- | --- | --- |
| DE-A | canonical raw6 no-PCM, ProPainter prior, d0, raw/no-comp | existing Gate32 reviewed |
| DE-B | OR-style preprocessing/dilation with same checkpoint | pending |
| DE-C | official PCM2 OR stack | pending |
| DE-D | official high-quality OR stack | pending |
| DE-E | official DiffuEraser core checkpoint | pending |
| DE-F | current SFT-48000 core checkpoint | pending |

Decision:

Do not label DiffuEraser OR capability failed yet. Current status is
`DIFFUSERASER_OR_PROTOCOL_ROOT_CAUSE_PENDING`.
