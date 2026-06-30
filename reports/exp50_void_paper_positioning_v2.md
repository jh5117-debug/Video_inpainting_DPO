# Exp50 VOID Paper Positioning V2

Time: 2026-07-01T00:11:20+08:00

Positioning status: `VOID_TRUE_ADAPTER_FEASIBLE_NEEDS_MICRO_TRAINING`

VOID remains a technically validated PAI inference baseline and a candidate loser generator for VOR-OR. It also has an isolated preference-forward wrapper with SFT parity explained, preference forward passed, and zero-gap passed.

Do not claim VOID as third adapter evidence. The H4 one-step checkpoint is valid, but H4b could not generate video-level heldout evidence because all PAI GPUs were occupied by unrelated root jobs. H5 10-step was not run.

Allowed wording:

- VOID is usable for official VOR-OR inference smoke on PAI.
- VOID can provide baseline outputs and some bounded loser candidates.
- VOID has passed engineering gates through preference forward and zero-gap.
- VOID still needs heldout video evidence for one-step before any 10-step micro-gate claim.

Forbidden wording:

- Do not claim a universal adapter.
- Do not claim final SOTA.
- Do not count VOID as third adapter evidence.
