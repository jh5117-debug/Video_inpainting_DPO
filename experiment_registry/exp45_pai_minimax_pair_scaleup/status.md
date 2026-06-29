# Exp45 Status

Current status: `MINIMAX_STAGE2_FORMAL_DATA_READY`

## 2026-06-29 Scope Correction And Readback

- Branch: `research/exp45-pai-minimax-pair-scaleup-20260629`
- Start HEAD: `81ad11ac08267fcc5db8bd0ebe9bd41bc9fca620`
- Exp44 source status: `MINIMAX_STAGE2_DATA_HANDOFF_PARTIAL`
- Existing split: `24/8/8`
- Formal minimum split: `32/16/16`
- Preferred split: `64/24/24`
- Bad-noise v4: `MINIMAX_BADNOISE_V4_READY`
- Training unlocked: `false`
- H20 touched by Exp45: `false`
- Training run by Exp45: `false`
- Optimizer step by Exp45: `false`
- Current session `/mnt/nas` mounted: `false`

Reports:

- `reports/exp45_pair_scaleup_readback.md`
- `reports/exp45_scope_deviation_h20_execution.md`

## 2026-06-29 H20 Handoff Filelist

- Status: `EXP45_H20_FILELIST_PARTIAL_SOURCE_ROOT_UNAVAILABLE`
- Scanned manifest rows: `120`
- Required absolute paths: `262`
- Visible absolute paths in this session: `0`
- Missing absolute paths in this session: `262`
- Repo-side manifests/reports checksummed: `16`
- PAI executed H20 mirror: `false`
- H20 touched by Exp45: `false`
- Training run by Exp45: `false`
- Optimizer step by Exp45: `false`

Reports:

- `reports/exp45_h20_required_filelist.txt`
- `reports/exp45_h20_required_sha256.txt`
- `reports/exp45_h20_required_filelist.csv`
- `reports/exp45_h20_handoff_package.md`
- `reports/exp45_h20_handoff_package.json`

## 2026-06-29 Targeted Pair Scale-Up Mining

- Status: `MINIMAX_TARGETED_MINING_COMPLETED`
- Existing pair count: `40`
- Existing split: `24/8/8`
- Minimum split target: `32/16/16`
- Preferred split target: `64/24/24`
- Real PAI host: `dsw-753014-85f54df947-bkp7h`
- Source root visible on PAI: `true`
- PAI run root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp45_pai_minimax_pair_scaleup/scaleup_mining_20260629_190229`
- New candidates mined: `72`
- automatic success candidates: `38`
- automatic medium-hard failure candidates: `26`
- automatic too-close candidates: `6`
- automatic fogging / over-erasure candidates: `2`
- automatic overlap groups: `5`
- MiniMax inference launched: `true`
- PAI GPU0/GPU1 used: `true`
- OOM/CUDA/Xid observed: `false`
- H20 touched by Exp45: `false`
- Training run by Exp45: `false`
- Optimizer step by Exp45: `false`
- VOR-Eval used: `false`
- hard comp used: `false`

Reports:

- `reports/exp45_targeted_scaleup_mining.md`
- `reports/exp45_targeted_scaleup_mining.csv`
- `reports/exp45_targeted_scaleup_summary.json`

## 2026-06-29 Strict Visual Relabel

- Status: `MINIMAX_TARGETED_RELABEL_COMPLETED`
- New candidates relabeled: `72`
- Selected auto success/failure rows reviewed: `64`
- Review pages generated: `8`
- Review pages inspected: `8`
- Accepted `SUCCESS_CLEAN`: `8`
- Accepted `SUCCESS_USABLE` including clean: `28`
- Accepted `FAILURE_MEDIUM_HARD`: `22`
- Rejected borderline: `14`
- Rejected fogging: `2`
- Rejected too-close: `6`
- Same-source groups with accepted pairs: `4`
- One-to-one same-source pair precheck from new rows: `8`
- Capped same-source combination precheck from new rows: `16`
- H20 touched by Exp45: `false`
- Training run by Exp45: `false`
- Optimizer step by Exp45: `false`
- VOR-Eval used: `false`
- hard comp used: `false`

Reports:

- `reports/exp45_visual_relabel.md`
- `reports/exp45_visual_relabel.csv`
- `reports/exp45_visual_relabel_summary.json`

## 2026-06-29 Formal Stage2 Handoff

- Status: `MINIMAX_STAGE2_FORMAL_DATA_READY`
- Pseudo-success split: `64/24/24`
- GT distillation split: `64/24/24`
- Same-source preference split: `64/24/24`
- Formal minimum split: `32/16/16`
- Preferred split: `64/24/24`
- Total same-source pair rows: `112`
- Scene-group overlap: `0`
- All PAI manifest path checks passed: `true`
- H20 handoff filelist status: `EXP45_H20_FILELIST_READY`
- H20 required paths: `326`
- H20 required missing paths: `0`
- H20 filelist total ready bytes: `99685806`
- Training status: `TRAINING_UNLOCKED_FOR_H20_HANDOFF`
- First H20 experiment: `pseudo-success SFT 30-step`
- Do not start first: `GT-only SFT`
- H20 touched by Exp45: `false`
- Training run by Exp45: `false`
- Optimizer step by Exp45: `false`
- VOR-Eval used: `false`
- hard comp used: `false`

Reports:

- `reports/exp45_stage2_formal_handoff.md`
- `reports/exp45_stage2_formal_handoff.csv`
- `reports/exp45_stage2_formal_handoff_summary.json`
- `reports/exp45_h20_handoff_instructions.md`

## 2026-06-29 Paper Positioning

- Paper status: `MINIMAX_DATA_SIGNAL_EMERGING_PAIR_YIELD_WEAK`
- MiniMax third adapter evidence: `false`
- Universal adapter language allowed: `false`
- Main positive adapter evidence remains: `DiffuEraser + VideoPainter`
- Next action: resume targeted mining from a true PAI/NAS-mounted session

Report:

- `reports/exp45_minimax_paper_positioning.md`

## 2026-06-29 HAL Environment Correction

- Status: `EXP45_HAL_ENVIRONMENT_BLOCKER_CORRECTION_RECORDED`
- Previous execution host: `hal-9000`
- Required `/mnt/nas` available in previous session: `false`
- New candidates mined: `0`
- New visual relabel rows: `0`
- Split remains: `24/8/8`
- Formal minimum met: `false`
- Real PAI C/D/E still required: `true`

Report:

- `reports/exp45_hal_environment_blocker_correction.md`
