# Exp40 MiniMax Paper Positioning

Status: `TWO_BACKBONE_PLUS_MINIMAX_PLUMBING_ONLY`

Exp40 tested whether the small Exp38 R1 raw-PSNR signal could be rescued by a PSNR-safe winner-SFT warmup before any DPO-after-SFT gate.

Result: `MINIMAX_SFT_PSNRSAFE_NEGATIVE`.

Key evidence:

- LocalDPO v3 minimum pool was ready: train64/search24/shadow24, VOR-Train only, zero scene overlap.
- Step0 baseline was established on train/search/shadow.
- SFT-A/B/C/D x LR 3e-5/1e-4/3e-4 was run for 30 steps on GPU0/GPU1.
- All 12 recipe aggregates failed the search numeric gate.
- Best aggregate recipe `SFTmC_S0_lr3em05` still had negative full/mask/boundary/outside deltas: `-1.816781` / `-1.634597` / `-1.899575` / `-2.624405`.
- Representative visual review showed isolated local changes but no recipe-level win, and high-LR settings produced noisy/color collapse.

Allowed paper language:

- DiffuEraser + VideoPainter remain the confirmed positive adapter evidence.
- MiniMax remains trainable and inference-sensitive, but the current objective/data recipe is not quality-positive.
- Exp40 rules out this PSNR-safe SFT grid as a path to a MiniMax third-backbone claim.

Forbidden paper language:

- `UNIVERSAL_ADAPTER`
- `ALL_MODELS_SUPPORTED`
- `FINAL_SOTA`
- `TOP_CONFERENCE_NOVELTY_CONFIRMED`
- MiniMax as third successful adapter evidence from Exp40.

Next minimal experiment:

Do not extend these recipes. If MiniMax is revisited, use a much smaller diagnostic that explicitly prevents full-frame drift, such as a frozen-region latent consistency test or architecture-specific low-rank adapter with per-layer output clamps, before any 30-step quality gate.
