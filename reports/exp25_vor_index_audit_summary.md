# Exp25 VOR Index / Pairing / Gate Summary

## Full Member Index

- VOR-Train members seen/written: 117978 / 117978
- VOR-Train-MASK members seen/written: 60225 / 60225
- path-safety unsafe members: 0
- combined unique member-path audit: 178203 unique paths, 986 duplicate rows from interrupted append attempts; downstream pairing uses unique member paths.
- full index CSV on PAI: reports/vor_train_mask_member_index.csv

## Triplet Pairing Rule

- Complete triplets: 57751 before semantic exclusion; 57750 after excluding failed audit sample BLENDER_RIVER007_00001.
- Pairing is exact video basename across roles:
  - OR condition: VOR-Train/FG_BG, i.e. V_obj.
  - winner: VOR-Train/BG, i.e. V_bg.
  - mask: VOR-Train-MASK/MASK.
- hard_comp: false. VOR-Eval is not used.

## Audit64

- selected: 64 triplets, source split REAL=40, BLENDER=24.
- extracted files: 192 / 192.
- semantic audit: 63 OK, 1 failed.
- failed sample: BLENDER_RIVER007_00001, reason frame_or_size_mismatch (FG_BG/MASK 11 frames, BG 240 frames).
- ffprobe binary is blocked on this PAI image by missing libblas/liblapack; audit used OpenCV fallback and records opencv_fallback_samples=64.
- visual spot check: REAL and BLENDER contact sheets confirm FG_BG condition, BG winner, and mask overlay are aligned.

## Group-Level Splits

- train source pool: 4096
- search-dev: 256
- shadow-dev: 256
- split grouping: scene_group/base-video isolation.
- group overlap counts: train-search=0, train-shadow=0, search-shadow=0.
- VOR-Eval 43 remains excluded from train/selection/threshold/checkpoint choice.

## Gate128

- Gate128 manifest: exp25_vor_or_preference_data/manifests/vor_gate128.jsonl
- source split: REAL=80, BLENDER=48.
- extracted files: 384 / 384.
- extraction status: ok=true, missing=0.
- ProPainter 32 list: exp25_vor_or_preference_data/manifests/vor_gate128_propainter_32.jsonl, REAL=20, BLENDER=12.
- EffectErase 32 list: exp25_vor_or_preference_data/manifests/vor_gate128_effecterase_32.jsonl, REAL=20, BLENDER=12.
- OR loser generation not started in this turn; no full 4608 extraction was performed.
