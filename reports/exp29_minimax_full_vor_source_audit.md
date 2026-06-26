# Exp29 MiniMax Full-VOR Source Audit

Status: `MINIMAX_FULL_VOR_SOURCE_AUDIT_READY`

## Inputs

- Full metadata index: `/mnt/nas/hj/H20_Video_inpainting_DPO_exp25_vor/exp25_vor_or_preference_data/manifests/vor_train_metadata_index.jsonl`
- Full metadata SHA256: `33d57a3ea23c5799b583d476a311089f95cbce1b0d11280822a63b8c9edcddc4`
- Previous source32 excluded rows: 32
- EffectErase smoke excluded rows: 12

## Counts

- Raw rows read: 57751
- Unique valid candidate scene groups after exclusions: 1417
- Selected source groups: 192
- Selected manifest SHA256: `16e128282da110eeefd6cb56a517c8b6de82e42a5241c9b845e01315d9800f10`

## Balance

- Selected source type counts: `{'REAL': 96, 'BLENDER': 96}`
- Mask buckets: `unknown_pending_materialization` because the full metadata index does not contain mask-area fields.
- Effect labels: `unknown_pending_metadata` because the full metadata index does not contain effect-type fields.
- Motion labels: `unknown_pending_metadata` because the full metadata index does not contain motion fields.

## Interpretation

This milestone does not claim MiniMax micro-data quality. It only fixes the
previous source-pool blocker by deriving a scene-disjoint candidate pool from
the full VOR Train metadata index. Mask non-emptiness, 17-frame decode,
medium-hard quality, and defect labels must be measured during the next
first-pass generation/materialization milestone before any training or recipe
gate is allowed.

No VOR-Eval rows are used, no MiniMax generation was launched, and no training
manifest was created by this audit.
