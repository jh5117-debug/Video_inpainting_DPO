# Exp29 EffectErase Inference Smoke V2

Date: 2026-06-26 / PAI 2026-06-27

Status: `EFFECTERASE_SMOKE_BLOCKED_FRAME_COUNT_INCOMPATIBLE`

## Scope

This milestone attempted the official EffectErase removal inference smoke on
the preregistered v2 six-row diagnostic manifest. It did not run training,
adapter gates, MiniMax generation, or any long training.

## Inputs

- Manifest:
  `exp29_or_adapter_feasibility/manifests/effecterase_smoke_preregistered_v2.jsonl`
- Manifest SHA256:
  `b16a0007a22f190bb7894a673092063efb5dd2eda26dbd53737cdc987d9d4f36`
- Rows: 6
- Resolution: 832x480
- Frames: 17
- Seed: 2025
- CFG: 1.0
- Inference steps: 50
- VOR-Eval use: false
- Training eligibility: false

## Runtime

- PAI hostname: `dsw-753014-85f54df947-bkp7h`
- GPU used: GPU0
- Runner PID: `594851`
- Runner PGID: `594851`
- Output root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_inference_smoke_v2_20260626`
- Runtime directory:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp29_or_adapter_effecterase_v2`
- Official repo path:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/third_party/EffectErase`
- License: `CC BY-NC 4.0`

## Attempts

Attempt 1 failed before model load because the official repo root was not on
`PYTHONPATH`, causing `ModuleNotFoundError: No module named 'diffsynth'`.
The log is preserved as:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_inference_smoke_v2_20260626/logs/REAL_ENV231_00010_003_03.attempt1_missing_diffsynth.log`

Attempt 2 added the EffectErase repo root to `PYTHONPATH`. The model, text
encoder, VAE, image encoder, and LoRA loaded successfully, and the official
pipeline reached inference. It then failed with:

`RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 21 but got size 5 for tensor number 1 in the list.`

The decisive log lines show:

- pipeline noise latent time dimension: `21`, which corresponds to 81 frames;
- condition/mask latent time dimension: `5`, which corresponds to the locked
  17-frame inputs.

Code audit confirms `infer_remove_wan.py` reads inputs with
`args.num_frames`, but does not pass `num_frames=args.num_frames` into
`WanRemovePipeline.__call__`, whose default is 81 frames.

## Decision

The official v2 smoke is blocked by frame-count incompatibility between the
locked 17-frame diagnostic inputs and the official pipeline default. Per the
pre-registered rule, this milestone does not patch the official script, does
not expand inputs to 81 frames, does not change the manifest, and does not
claim baseline readiness.

Final status for this milestone:

`EFFECTERASE_SMOKE_BLOCKED_FRAME_COUNT_INCOMPATIBLE`

No EffectErase output video was produced, so no video quality, metric, OR
baseline-ready, adapter-ready, or scientific-positive claim is supported.
