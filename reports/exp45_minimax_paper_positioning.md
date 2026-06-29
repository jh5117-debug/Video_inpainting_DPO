# Exp45 MiniMax Paper Positioning

Status: `MINIMAX_DATA_SIGNAL_EMERGING_PAIR_YIELD_WEAK`

## Outcome

Exp45 did not reach the formal Stage2 data gate. The final handoff remains:

- pseudo-success distillation: `24/8/8`
- GT distillation: `24/8/8`
- same-source preference: `24/8/8`
- formal minimum: `32/16/16`
- preferred target: `64/24/24`
- training status: `TRAINING_NOT_UNLOCKED`

The blocker is operational/data-access, not a new MiniMax quality result:
the current session cannot access `/mnt/nas` or `/mnt/workspace`, so PAI
targeted mining could not run and no new candidates could be visually reviewed.

## Allowed Language

- MiniMax has curated, human-verified same-source success/failure signal from
  Exp44.
- MiniMax has bad-noise v4 data artifacts from Exp44.
- Exp45 produced an indexed partial handoff and H20 filelist package.
- Exp45 confirms that more PAI/NAS-mounted mining is required before formal
  Stage2 training should be unlocked.

## Forbidden Language

- MiniMax third-backbone positive.
- MiniMax adapter success.
- Universal adapter.
- All models supported.
- Final SOTA.
- Top-conference novelty confirmed.

## Scientific Position

DiffuEraser and VideoPainter remain the main positive adapter evidence. MiniMax
remains a protocol-audited, plumbing-positive, data-signal-emerging extension.
It must not be counted as a third successful adapter until a later H20 or PAI
training run passes search/shadow quality gates with real video review.

## Next Minimal Experiment

Resume Exp45 Milestone C from a true PAI/NAS-mounted session:

1. verify `/mnt/nas` and `/mnt/workspace` are mounted;
2. verify Exp44 targeted mining source root is readable;
3. use GPU0/GPU1 only if assigned and free;
4. mine success-only, failure-only, and overlap groups with the preregistered
   budget;
5. strict visual relabel;
6. rebuild the Stage2 split to at least `32/16/16`;
7. only then hand H20 a formal pseudo-success SFT 30-step package.
