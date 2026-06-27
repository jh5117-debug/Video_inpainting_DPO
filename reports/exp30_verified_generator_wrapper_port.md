# Exp30 Verified Generator Wrapper Port

Date: 2026-06-27

Status: `EXP30_VERIFIED_GENERATOR_WRAPPERS_PORTED_SMOKE2_PENDING`

## Readback

Before this milestone, the worktree was on
`research/exp30-vor-or-multimodel-minimax-adapter-20260627` at
`bce81988da21adf6640142bb669b94aa9bf2a4ca`, with clean local status.

Read files:

- `PRD/50_exp30_vor_or_multimodel_minimax.md`
- `experiment_registry/exp30_vor_or_multimodel_minimax/status.md`
- `reports/exp30_diffueraser_propainter_candidate_audit.md`
- `exp30_vor_or_multimodel_minimax/scripts/run_controlled_corruption_smoke16_v2.py`
- Exp25 verified DiffuEraser no-PCM overlay wrapper source, read-only

Protected-lane readback:

- PAI GPU1: Exp31/cli4 process observed.
- PAI GPU3: non-Exp30 EffectErase process observed.
- cli4 locks still reserve GPU1-GPU4.
- No signal was sent and no protected file was modified.

## Changes

Added Exp30-isolated copies of the verified generator wrappers:

- `exp30_vor_or_multimodel_minimax/scripts/infer_diffueraser_or_exp30.py`
- `exp30_vor_or_multimodel_minimax/scripts/run_verified_or_generator_smoke.py`

The DiffuEraser wrapper preserves the Exp25 verified identity pattern:

- explicit `--pcm_mode`
- explicit no-PCM mode
- no silent PCM fallback
- overlay-only `diffueraser_OR.py` patch
- overlay-only `run_OR.py` seed forwarding
- raw/no-comp output frames

Exp30-specific changes:

- wrapper identity now records
  `exp30_vor_or_multimodel_minimax/scripts/infer_diffueraser_or_exp30.py`
- no-PCM environment variables are `EXP30_NO_PCM_STEPS` and
  `EXP30_NO_PCM_GUIDANCE`
- smoke runner supports only `diffueraser` and `propainter`
- smoke runner default frames are 17 to match the repaired Smoke16 v2
  materialized inputs
- smoke runner default DiffuEraser checkpoint path is
  `/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000`
- ProPainter default weight path is
  `/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter`

## Not Run

No GPU smoke was launched in this milestone.

Still pending:

- create/sync Exp30 PAI worktree;
- run two-sample DiffuEraser no-PCM smoke;
- run two-sample ProPainter smoke;
- review all smoke videos;
- only then enable these families in Smoke16 v3.

Still stopped:

- Smoke16 v3;
- Smoke32;
- Gate64;
- MiniMax adapter gate;
- DiffuEraser training/micro;
- RC-FPO;
- long training.

