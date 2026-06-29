# Exp44 Status

Current status: `EXP44_TARGETED_SOURCE_MANIFEST_READY`

## 2026-06-29 Readback

- Branch: `research/exp44-pai-minimax-targeted-same-source-mining-20260629`
- Base: `origin/research/exp42-pai-minimax-successful-removal-badnoise-20260629`
- Start HEAD: `efc07b2e9f5e84fe488b433b812a9f0bc72debaf`
- Exp42 overlap: `7` same-source success/failure scene groups.
- Plan locked: `56` total target groups across A/B/C/D.
- PAI GPU0/GPU1 were occupied by unrelated root-owned
  `qxq_sample_valtest_v0.py` jobs during readback; no cleanup was attempted.
- No GPU mining, training, optimizer step, H20 action, VOR-Eval use, or hard comp launched.

Reports:

- `reports/exp44_minimax_targeted_readback.md`
- `reports/exp44_source_group_plan.csv`
- `reports/exp44_source_group_plan.json`

## 2026-06-29 Targeted Source Manifest

- Status: `EXP44_TARGETED_SOURCE_MANIFEST_READY`
- Source rows: `40`
- Candidate budget: `452`
- Missing source rows: `0`
- Manifest: `exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_targeted_source_manifest.jsonl`
- Manifest SHA256: `5147839e1e2d60e0ecc9c77a438a934918605b5fa550fa58d1e3291df7be168b`
- Runner: `exp44_pai_minimax_targeted_same_source_mining/scripts/mine_targeted_candidates.py`

No GPU inference or training has been launched by this preparation step.
