# Exp49 ROSE Paper Positioning

ROSE is not counted as third adapter evidence.

Allowed wording:

- ROSE official inference is technically runnable on PAI with downloaded Wan/ROSE weights.
- ROSE has a VOR-Train baseline / loser-generator signal: Gate16 is `ROSE_VOR_OR_GATE16_PASS` with `16/16` decodable outputs and `14/16` useful baseline or medium-hard loser candidates after visual review.
- ROSE is worth continued isolated feasibility work because its official inference can remove objects on VOR-like rows and produce bounded side-effect failures.

Forbidden wording:

- Do not claim `ROSE_ADAPTER_POSITIVE`.
- Do not claim ROSE is the third positive backbone/adapter.
- Do not claim universal adapter or final SOTA.
- Do not claim trainability from inference-only evidence.

Current paper-safe conclusion:

DiffuEraser + VideoPainter remain the positive adapter evidence. MiniMax remains negative/not third-backbone based on Exp46/Exp47-style outcomes, and ROSE is a promising baseline/loser-generator candidate but still adapter-blocked until a true trainable objective passes zero-gap, one-step, reload, and heldout micro gates.
