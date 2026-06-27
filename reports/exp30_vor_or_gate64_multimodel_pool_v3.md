# Exp30 Gate64 Multi-Model OR Pool V3

Status: `VOR_OR_GATE64_MULTIMODEL_POOL_READY`

- Candidate rows: 256
- Source rows: 64
- Selected primary pairs: 50
- Train split: 32
- Heldout split: 16
- Train/heldout scene overlap: `[]`
- Selected model counts: `{'controlled_corruption_v3': 26, 'diffueraser': 1, 'minimax_official_v3': 17, 'propainter': 6}`
- Selected class counts: `{'HARD_BUT_PLAUSIBLE': 2, 'MEDIUM_HARD_ELIGIBLE': 48}`
- Train model counts: `{'controlled_corruption_v3': 15, 'diffueraser': 1, 'minimax_official_v3': 10, 'propainter': 6}`
- Heldout model counts: `{'controlled_corruption_v3': 11, 'minimax_official_v3': 5}`
- Candidates SHA256: `a64610e369e2904de5331913501a55736bc71d962644663e8ac0869dee1568a1`
- Selected primary SHA256: `f1621a7fa5dd61844521a7bbfcc6dd45ad0378db4bf0440f8afe787558c3f4c3`
- Train32 SHA256: `1eda205d2dc48714269f30eb390d959549387a778e6438267e6aba087ba14196`
- Heldout16 SHA256: `84c231ded930d740bf299b27c2a6b1e95d7decdb3665051371c5df90ae9f2ade`

The pool uses raw OR losers only.  EffectErase is not included as a primary loser.  No training is launched by this aggregation step.
