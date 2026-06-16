# Exp15 OR Method Runtime Status

| Method | Status | Evidence / entry | Notes |
|---|---|---|---|
| ProPainter | COMPLETED_50_50 | `DPO_finetune/infer_propainter_candidate.py` | Uses existing ProPainter weights on PAI. |
| VideoComposer / VideoComp | BLOCKED_NO_REPO | none verified | No verified PAI repo+weights+OR wrapper. |
| CoCoCo | BLOCKED_NO_WEIGHT | `DPO_finetune/infer_cococo_candidate.py` | COCOCO repo/checkpoints exist, but the SD inpainting dependency is incomplete. |
| FloED | BLOCKED_NO_REPO | none verified | No verified PAI repo+weights+OR wrapper. |
| DiffuEraser SFT-48000 | COMPLETED_50_50 | `DPO_finetune/infer_diffueraser_candidate.py` | Uses SFT-48000 weights. |
| VideoPainter | BLOCKED_NO_OR_WRAPPER | Exp14 BR wrapper exists | No verified DAVIS2017 foreground-mask OR wrapper for this benchmark. |
| VACE | BLOCKED_NO_REPO | none verified | No verified PAI repo+weights+OR wrapper. |
| Ours Exp11 outer b0.75 S2 | COMPLETED_50_50 | `DPO_finetune/infer_diffueraser_candidate.py` | Uses current best Exp11 boundary outer b0.75 S2 weights. |
| MiniMax-Remover | BLOCKED_IMPORT_ERROR | `DPO_finetune/infer_minimax_candidate.py` | Real repo/weights exist, but it needs an isolated env with newer diffusers. |

Only verified runnable methods were executed in the DAVIS50 run. Blocked methods
remain explicit placeholders in visual grids and result tables.
