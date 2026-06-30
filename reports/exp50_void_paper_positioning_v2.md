# Exp50 VOID Paper Positioning V2

Time: 2026-07-01T01:07:56+08:00

Positioning status: `VOID_ONE_STEP_VIDEO_EVIDENCE_MIXED`

VOID remains a technically validated VOR-OR inference baseline and candidate loser generator. It also has an isolated preference-forward wrapper that passes SFT parity, preference forward, and zero-gap.

The one-step heldout video evidence is mixed. Two heldout samples were visual ties and two were worse/mixed by metrics or review. Mean full PSNR delta was -0.025049, which fails the `>= -0.02` one-step PASS threshold. H5 10-step was not run.

Allowed wording:

- VOID is usable for VOR-OR inference smoke on PAI.
- VOID can provide baseline outputs and bounded loser candidates.
- VOID preference-forward engineering is feasible.
- VOID one-step video evidence is mixed and does not unlock 10-step.

Forbidden wording:

- Do not claim a universal adapter.
- Do not claim final SOTA.
- Do not count VOID as third adapter evidence.
