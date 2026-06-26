# Exp29 EffectErase Official 81F Source Audit

Status: `EFFECTERASE_OFFICIAL81_PREREGISTERED`

## Inputs

- Metadata index: `/mnt/nas/hj/H20_Video_inpainting_DPO_exp25_vor/exp25_vor_or_preference_data/manifests/vor_train_metadata_index.jsonl`
- Metadata SHA256: `33d57a3ea23c5799b583d476a311089f95cbce1b0d11280822a63b8c9edcddc4`
- Extraction CSVs: `['/mnt/nas/hj/H20_Video_inpainting_DPO_exp25_vor/.tmp/pre_exp25_ff_20260623_121851/reports/vor_gate128_exact_extraction.csv', '/mnt/nas/hj/H20_Video_inpainting_DPO_exp25_vor/.tmp/pre_exp25_ff_20260623_121851/reports/vor_triplet_audit64_exact_extraction.csv']`
- VOR-Eval use: false
- Training eligibility: false

## Counts

- Candidate triplets audited from exact extraction caches: 24
- Accepted by 81F/frame/mask rules: 24
- Manifest rows locked: 8
- Rejected rows recorded: 16
- Manifest SHA256: `706cb09286fd8528d7efbbb91eb89673a9ec7ce61b0047e6b3b2e8ea4c9b1fb3`
- Rejected manifest SHA256: `7700a99c5585d4b8759527bd48029b6ad90a8ca8c3c304a877b0bc1fbcce0f6e`
- Source type counts: `{'REAL': 5, 'BLENDER': 3}`
- Mask bucket counts: `{'small': 3, 'medium': 3, 'large': 2}`

## Protocol

- Selected frames: exact consecutive frame indices 0-80.
- Frame count requirement: condition/winner/mask each have at least 81 decoded frames.
- Mask requirement: at least 40/81 non-empty frames and median mask area in [0.001, 0.60].
- Primary output remains raw EffectErase output; diagnostic comp is optional later.
- Rows are diagnostic-only VOR-confounded rows and are not eligible for training.

## Preview Review

Preview sheets were generated for each locked row under:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_official81_source_audit_20260627/reports/exp29_effecterase_official81_previews`

They contain condition, winner, and mask-overlay strips across 16 sampled frames.
Codex visual opening is recorded separately in `exp29_effecterase_official81_preview_review.csv`.

## Codex Preview Review

Codex opened all 8 generated preview sheets on 2026-06-27. The selected rows
show continuous 81-frame source motion, non-empty masks, aligned
condition/winner pairs, and no visible padding, loop, empty-mask, or frame-order
failure in the sampled strips. The locked manifest remains diagnostic-only and
does not unlock training or adapter claims.
