# Exp25 DiffuEraser OR Root-Cause Matrix Status - 2026-06-25

Status: `BLOCKED_WEIGHT_PERMISSION_BEFORE_STACK_COMPARISON`

A fixed 12-sample root-cause manifest was constructed from Gate32 dense review evidence:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_vor_or_preference_data/root_cause_matrix_20260625/root_cause_sample12_manifest.jsonl`

SHA256: `d1a7ef848ce1f5777ae80f1655c581fa5328d108fab497693d8afddf750afa49`

Bucket coverage from available Gate32 rows:

- medium-hard REAL: 1
- medium-hard BLENDER: 6
- trivial-bad REAL: 3
- trivial-bad BLENDER: 2

Attempted runnable stacks:

- DE-B: no PCM, ProPainter prior, mask dilation 8
- DE-C: official PCM2, ProPainter prior, mask dilation 8

DE-B failed before model inference on all 12 rows because the current SSH user `hj` cannot read the DiffuEraser checkpoint directory:

`/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000`

The specific error is `PermissionError: [Errno 13] Permission denied` when reading `brushnet/config.json` / `unet_main/config.json`. Root SSH is not available with the current key, and no readable duplicate checkpoint was confirmed in this run.

This is an infrastructure permission blocker, not a DiffuEraser OR quality result. The root-cause matrix remains pending; no Gate128 expansion and no OR-DPO training were launched.

Required fix: from PAI root terminal, grant `hj` read/execute access to the DiffuEraser, SD1.5, VAE, PCM, and ProPainter weight directories or provide a readable copy under an Exp25/NAS runtime asset path.
