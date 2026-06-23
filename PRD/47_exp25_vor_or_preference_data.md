# Exp25 VOR OR Preference Data

- repo: FudanCVL/EffectErase
- HF authenticated user: JiaHuang01
- dataset revision: `fa09dc61128ca0418a4a13364d97a08018ea9cc7`
- PAI outbound HF network: unavailable; use HAL-only download then rsync to PAI.
- core scope: README, VOR-Eval parts, VOR-Train-MASK parts, VOR-Train parts.
- excluded this round: VOR-Wild.
- required files: 37
- required total bytes: 363730944386
- largest part bytes: 10737418240
- HAL staging: `/home/hj/exp25_effecterase_staging`
- HAL free bytes at selection: 571536965632
- PAI destination: `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/downloads/fa09dc61128ca0418a4a13364d97a08018ea9cc7`
- status: GATE128_EXTRACTED
- status_detail: THREE_MODEL_SMOKE_PARTIAL_BLOCKED
- completed time: 2026-06-23T00:08:45+0200
- completed files: 37 / 37
- completed bytes: 363730944386 / 363730944386
- PAI final files: 37
- PAI partial files: 0
- PAI bad files: 0
- HAL staging final size: 1.0K

## Safety

This track is download-only. It does not enter Exp23 worktrees, use GPUs, run inference, generate losers, or start DPO training. Tokens remain only under `/home/hj/.cache/huggingface_effecterase_auth` and are not copied to PAI or committed.

## Completion Notes

The core EffectErase VOR compressed archive scope completed via HAL-only Hugging Face download and HAL-to-PAI rsync. Each file was processed serially, verified with HAL and PAI SHA256 equality before atomic finalization, and the per-file HAL job/cache was removed after verification.

The completion marker exists on PAI:

`/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/runtime/CORE_DOWNLOAD_COMPLETE`

Final PAI-side inventory verification completed successfully after transfer. The independent verifier checked all 37 required files under the fixed revision directory, reported `ok=true`, found 0 partial files and 0 bad files, and confirmed contiguous archive parts for VOR-Eval, VOR-Train-MASK, and VOR-Train. The append-only transfer manifest also contains 37 VERIFIED rows and zero HAL/PAI SHA256 mismatches from transfer-time checks.

## Next Phase: Selective OR Data Construction

The next phase must not materialize or generate losers for the full 60K VOR
training set. Exp25 now adds isolated tooling for:

- lightweight split-archive continuity and byte-count inspection;
- resumable tar member indexing with path-safety checks;
- selective extraction of VOR-Eval and chosen VOR-Train/VOR-Train-MASK sample
  IDs;
- validation of extracted subsets;
- canonical OR manifest semantics where `condition = V_obj`, `winner = V_bg`,
  `mask = foreground object mask`, `hard_comp = false`, and losers are raw
  generator outputs.

The first formal source pool remains capped at 4096 train candidate triplets,
256 search-dev triplets, and 256 shadow-dev triplets. Preference manifests must
be nested at 512/1024/2048/3072, with 4096 allowed only if 3072 remains clearly
better than 2048.

## Archive Audit Result

PAI lightweight archive audit passed on the fixed revision directory. VOR-Eval
has 1/1 expected part, VOR-Train-MASK has 3/3 expected parts, and VOR-Train has
32/32 expected parts with contiguous numeric part IDs 000-031. Expected and
actual byte counts match for all three groups. A stream probe opened each
split gzip/tar stream and inspected the first five members with zero unsafe
paths found.

VOR-Eval was then fully extracted as the held-out final evaluation split. The
extracted `BG`, `FG_BG`, and `MASK` directories each contain 43 mp4 files, and
their basenames match exactly. This establishes the OR triplet semantics:
`condition=FG_BG/V_obj`, `winner=BG/V_bg`, and `mask=MASK`.

## 2026-06-23 Gate128 OR Smoke

Gate128 extraction is complete and a balanced 6-sample smoke set was materialized
as exact 24-frame inputs. ProPainter generated valid raw OR losers for all six
smoke samples with `hard_comp=false`.

DiffuEraser did not pass smoke: all six samples failed before generation because
the current DiffuEraser OR wrapper attempts to load PCM LoRA weights through a
diffusers API path that calls `UNetMotionModel.load_lora_adapter`, which is not
available in the active PAI environment. This is an inference-wrapper
compatibility blocker, not a VOR data issue.

EffectErase did not run because Exp25 still lacks a verified EffectErase
inference wrapper/checkpoint path. No fallback model was used.

Full Gate128 loser generation remains blocked until DiffuEraser and EffectErase
smoke pass under verified, no-fallback wrappers.

## 2026-06-23 DiffuEraser / EffectErase Inference Stack Audit

Current normalized status:

- `GATE128_EXTRACTED`
- `THREE_MODEL_SMOKE_PARTIAL_BLOCKED`

DiffuEraser clarification:

- The SFT/DPO core model is not a LoRA checkpoint. It is the full DiffuEraser
  `brushnet/` and `unet_main/` checkpoint.
- The current OR smoke failure is in the PCM inference acceleration adapter:
  `inference/run_OR.py` selects `ckpt = "2-Step"`, and
  `diffueraser/diffueraser_OR.py` loads
  `pcm_sd15_smallcfg_2step_converted.safetensors` through diffusers'
  `load_lora_weights(...)` path.
- This blocker is therefore
  `AUDIT_AND_FIX_DIFFUERASER_OR_PCM_INFERENCE_COMPATIBILITY`, not
  `FIX_DIFFUERASER_LORA_TRAINING`.

Two DiffuEraser OR stack identities are now separated:

- `DE_OFFICIAL_PCM2`: official 2-step PCM stack, blocked pending an
  official-pinned environment smoke.
- `DE_CANONICAL_NO_PCM`: policy-matched diagnostic candidate, verified
  historically for BR/DAVIS no-PCM but not yet promoted for OR/VOR.

EffectErase official repo was cloned and audited at commit
`bcee0a5da5ef387c2ba39390dc4d579503669fb8` under
`/home/hj/video_inpainting_third_party/EffectErase`. Official assets
`EffectErase.ckpt` and `Wan2.1-Fun-1.3B-InP` were not found locally, so EE-L0 is
still blocked by missing official weights/base model.

Reports:

- `reports/exp25_diffueraser_or_stack_audit.md`
- `reports/exp25_diffueraser_primary_stack_decision.md`
- `reports/exp25_effecterase_official_deployment_audit.md`
