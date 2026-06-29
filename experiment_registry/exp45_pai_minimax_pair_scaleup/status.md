# Exp45 Status

Current status: `MINIMAX_STAGE2_FORMAL_DATA_PARTIAL`

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

- Status: `MINIMAX_TARGETED_SCALEUP_BLOCKED_SOURCE_ROOT_UNAVAILABLE`
- Existing pair count: `40`
- Existing split: `24/8/8`
- Minimum split target: `32/16/16`
- Preferred split target: `64/24/24`
- Source root visible: `false`
- New candidates mined: `0`
- MiniMax inference launched: `false`
- H20 touched by Exp45: `false`
- Training run by Exp45: `false`
- Optimizer step by Exp45: `false`

Reports:

- `reports/exp45_targeted_scaleup_mining.md`
- `reports/exp45_targeted_scaleup_mining.csv`
- `reports/exp45_targeted_scaleup_summary.json`

## 2026-06-29 Strict Visual Relabel

- Status: `MINIMAX_TARGETED_RELABEL_BLOCKED_NO_CANDIDATES`
- New candidates: `0`
- Review pages generated: `0`
- Review pages inspected: `0`
- Accepted success: `0`
- Accepted failure: `0`
- H20 touched by Exp45: `false`
- Training run by Exp45: `false`
- Optimizer step by Exp45: `false`

Reports:

- `reports/exp45_visual_relabel.md`
- `reports/exp45_visual_relabel.csv`
- `reports/exp45_visual_relabel_summary.json`

## 2026-06-29 Formal Stage2 Handoff

- Status: `MINIMAX_STAGE2_FORMAL_DATA_PARTIAL`
- Pseudo-success split: `24/8/8`
- GT distillation split: `24/8/8`
- Same-source preference split: `24/8/8`
- Formal minimum split: `32/16/16`
- Preferred split: `64/24/24`
- Scene-group overlap: `0`
- Training status: `TRAINING_NOT_UNLOCKED`
- H20 touched by Exp45: `false`
- Training run by Exp45: `false`
- Optimizer step by Exp45: `false`

Reports:

- `reports/exp45_stage2_formal_handoff.md`
- `reports/exp45_stage2_formal_handoff.csv`
- `reports/exp45_stage2_formal_handoff_summary.json`
- `reports/exp45_h20_handoff_instructions.md`
