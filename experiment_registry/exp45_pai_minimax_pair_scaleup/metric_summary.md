# Exp45 Metric Summary

## Readback

Source Exp44 metrics:

- same-source usable pairs: `40`
- current split: `24/8/8`
- formal minimum split: `32/16/16`
- preferred split: `64/24/24`
- bad-noise v4 state records: `40`
- usable H-state records: `26`
- local/random gradient-proxy median ratio: `2.280567`
- scene overlap in current split: `0`

No Exp45 mining or training metrics have been produced yet.

## H20 Handoff Filelist

- manifest rows scanned: `120`
- required absolute paths: `262`
- visible absolute paths in current session: `0`
- missing absolute paths in current session: `262`
- visible raw/evidence bytes: `0`
- repo-side manifests/reports checksummed: `16`
- status: `EXP45_H20_FILELIST_PARTIAL_SOURCE_ROOT_UNAVAILABLE`

## Targeted Pair Scale-Up Mining

- new candidates mined: `72`
- automatic success candidates: `38`
- automatic medium-hard failure candidates: `26`
- automatic too-close candidates: `6`
- automatic fogging / over-erasure candidates: `2`
- auto overlap groups: `5`
- auto same-source pair capacity from new candidates alone: `16`
- full PSNR mean / min / max: `32.283717 / 26.747693 / 36.263905`
- mask PSNR mean / min / max: `25.240900 / 14.300294 / 32.359744`
- boundary PSNR mean / min / max: `24.419417 / 16.410627 / 28.990691`
- outside PSNR mean / min / max: `33.194681 / 27.709829 / 37.731807`
- temporal diff MAE mean / min / max: `1.377766 / 0.503339 / 3.632224`
- MiniMax inference launched on PAI: `true`
- H20 touched: `false`
- training / optimizer step: `false`
- VOR-Eval / hard comp: `false`
- status: `MINIMAX_TARGETED_MINING_COMPLETED`

## Strict Visual Relabel

- new candidates relabeled: `72`
- selected automatic success/failure rows reviewed: `64`
- review pages inspected: `8`
- accepted `SUCCESS_CLEAN`: `8`
- accepted `SUCCESS_USABLE` including clean: `28`
- accepted `FAILURE_MEDIUM_HARD`: `22`
- rejected borderline: `14`
- rejected fogging: `2`
- rejected too-close: `6`
- same-source groups with accepted pairs: `4`
- one-to-one same-source pair precheck from new rows: `8`
- capped same-source combination precheck from new rows: `16`
- status: `MINIMAX_TARGETED_RELABEL_COMPLETED`

## Formal Stage2 Handoff

- pseudo-success split: `24/8/8`
- GT distillation split: `24/8/8`
- same-source preference split: `24/8/8`
- formal minimum met: `false`
- scene overlap: `0`
- status: `MINIMAX_STAGE2_FORMAL_DATA_PARTIAL`
