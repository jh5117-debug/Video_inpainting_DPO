# Weights Directory

This directory should contain only lightweight placeholders, README files, or symlinks. Do not commit model checkpoints.

Recommended layout:

- `diffueraser/`
- `propainter/`
- `cococo/`
- `minimax_remover/`
- `official_videodpo/`
- `vc2/`

Use the environment variables in `configs/paths/pai.example.env` to point scripts at real checkpoint locations.

On PAI, run:

```bash
bash scripts/pai_audit_and_prepare_assets.sh
source configs/paths/pai.detected.env
```

The audit creates `current` symlinks under each model directory when the
corresponding asset is confirmed.
