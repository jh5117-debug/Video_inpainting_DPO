# Exp51 VOID Paper Positioning

Allowed:

- VOID official inference works on VOR-OR and can serve as an audited baseline.
- VOID can produce same-model loser candidates for preference experiments.
- VOID has a validated preference-forward / zero-gap / one-step foundation from Exp50.
- Exp51 identified a loser-dominant failure mode and isolated quadmask/local-region risks.

Not allowed:

- VOID third-backbone evidence.
- Universal adapter.
- Final SOTA.

Current wording:

VOID remains an adapter-engineering candidate and baseline/loser generator. It is not counted with DiffuEraser/VideoPainter positive adapter evidence until a heldout 10-step or larger gate is genuinely positive.
