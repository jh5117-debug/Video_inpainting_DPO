# Exp44 Stage2 Dataset Handoff

- Status: `MINIMAX_STAGE2_DATA_HANDOFF_PARTIAL`
- Training status: `TRAINING_NOT_UNLOCKED`
- Pair counts train/search/shadow: `24` / `8` / `8`
- Minimum train/search/shadow: `32` / `16` / `16`
- Preferred train/search/shadow: `64` / `24` / `24`
- Scene overlap train/search: `0`
- Scene overlap train/shadow: `0`
- Scene overlap search/shadow: `0`
- Bad-noise states matched to pairs: `40`
- Usable local/random ratio states: `28`

## View Counts

| view | train | search | shadow |
| --- | ---: | ---: | ---: |
| GT distillation | 24 | 8 | 8 |
| pseudo-success distillation | 24 | 8 | 8 |
| same-source preference | 24 | 8 | 8 |

## Interpretation

The handoff is partial because the same-source pair pool passes the Exp44
minimum pair gate but does not reach the dataset minimum
train32/search16/shadow16. H20 may use these manifests for a
bounded debug/preflight run, but this package must not be described as
formal training-unlocked evidence.

The first H20 experiment should be pseudo-success SFT 30-step. Do not
start with GT-only SFT first.

Path existence could not be validated in this Codex session because `/mnt/nas` is not mounted. H20 must verify every manifest path before running the dataloader or runner.

No training, optimizer step, VOR-Eval use, hard comp, or model update
occurred in this milestone.
