# Exp45 PAI MiniMax Pair Scale-Up

PAI-only data scale-up lane for MiniMax Stage2 handoff packaging.

Primary goal: expand the Exp44 partial `24/8/8` same-source handoff to at least
`32/16/16`, preferably `64/24/24`, without running training or touching H20.

Current status: `MINIMAX_STAGE2_FORMAL_DATA_READY`.

Final PAI handoff:

- new candidates mined: `72`
- accepted success usable including clean: `28`
- accepted medium-hard failures: `22`
- final formal split: `64/24/24`
- scene overlap: `0`
- H20 filelist: `EXP45_H20_FILELIST_READY`
- H20 touched by PAI: `false`
- training / optimizer step: `false`
