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

## 2026-06-23 DiffuEraser Stack v2

Exp25 now separates PCM and prior mode explicitly.

Main candidates:

- `DE_OFFICIAL_PCM2_PROP_PRIOR`: PCM2 acceleration plus ProPainter prior;
  official accelerated baseline, not strict on-policy.
- `DE_NO_PCM_PROP_PRIOR`: no PCM plus ProPainter prior; strict on-policy
  self-model loser candidate.
- `DE_NO_PCM_NO_PRIOR_DIAGNOSTIC`: not implemented as primary; changes prior
  variable and remains diagnostic-only.

New isolated wrapper:

`exp25_vor_or_preference_data/scripts/infer_diffueraser_or_exp25.py`

It only patches the temporary overlay copy of `diffueraser_OR.py`; shared code
and old Exp15 code are not modified.

Primary stack remains pending fixed smoke and visual review.

## 2026-06-23 DiffuEraser no-PCM Smoke6

PAI runtime snapshot:

`/mnt/workspace/hj/nas_hj/runtime_code_snapshots/exp25_934ec73`

Result:

- `DE_NO_PCM_PROP_PRIOR` generated the fixed Smoke6 set successfully.
- Generator ID: `diffueraser_or_none_propainter_62d00ca9c76a`.
- Decode/frame result: `6/6`, 24 frames each.
- Hard comp: `false`.
- VOR-Eval: not used.

Important environment fix:

The first probe failed because PyAV was missing. Installing public package
`av==17.1.0` fixed the torchvision video I/O dependency. This does not change
the model checkpoint or PCM mode.

Visual decision:

`DIFFUERASER_NO_PCM_TECHNICAL_PASS`

`READY_GATE128 = false`

Reason: the wrapper is technically valid, but loser utility is not yet strong
enough. The fixed Smoke6 review found only one clearly medium-hard eligible
case, several too-close/easy outputs, and one residual-artifact case.

Report:

`reports/exp25_smoke6_final_decision.md`

## 2026-06-23 No-PCM Canonical Identity Correction

The existing Smoke6 generator
`diffueraser_or_none_propainter_62d00ca9c76a` is now classified as an
OR-style no-PCM technical diagnostic, not the project raw6 canonical identity.
The smoke launcher passed `mask_dilation_iter=8`, while the verified
DAVIS/BR raw6 protocol uses no PCM, `6` UniPC steps, guidance `0.0`, and
`mask_dilation_iter=0`.

Exp25 now locks the canonical identity in:

`exp25_vor_or_preference_data/configs/diffueraser_or_canonical_no_pcm.json`

The smoke launcher and wrapper were fixed to record `mask_dilation_iter` in
the generator identity. Gate32/Gate128 remain blocked until a fresh canonical
Smoke6 with `mask_dilation_iter=0` passes technical and visual review.

Report:

`reports/exp25_no_pcm_canonical_identity_audit.md`

## 2026-06-23 Overnight GPU2 Gate32 Completion

The overnight Exp25/26/27 controller was resumed after GPU2 became free. A
controller scheduling bug was fixed: completed GPU subprocesses that remained
as zombies are no longer treated as alive.

Commit:

`22f7c54f49c46655700d7cb4193ac18dfa3bf037`

Exp25 Gate32 canonical DiffuEraser generation completed:

- Materialization: `32/32`.
- DiffuEraser raw OR candidates: `32/32` OK.
- Frames: `24` each.
- Generator ID: `diffueraser_or_none_propainter_abd3ad48f60f`.
- `pcm_mode=none`, `prior_mode=propainter`, `hard_comp=false`.
- Output:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/preference_candidates/gate32_canonical_raw6_d0`.

This remains Gate32 yield calibration only. No long training was started, and
VOR-Eval was not used.

Report:

`reports/exp25_overnight_gpu2_gate32_completion.md`

## 2026-06-24 Gate32 Yield Review

The three-lane controller completed the Exp25 Gate32 yield review without
starting Gate128 or any training. All 32 canonical DiffuEraser raw OR
candidates were scored and visually checked through contact-sheet indices.

Bucket counts:

- `medium-hard`: 11
- `too-close`: 0
- `trivial-bad`: 21

Because no `too-close` samples were found, seed2 supplementation was not
launched. The medium-hard yield is not strong enough to justify immediate
Gate128 expansion. Report:

`reports/exp25_gate32_yield_review_20260624.md`

## 2026-06-24 Individual Gate32 Reaudit

Status:

- `GATE32_GENERATION_PASSED`
- `GATE32_QUALITY_YIELD_POOR`
- `INDIVIDUAL_VIDEO_REAUDIT_FRAME_SAMPLE_COMPLETE_PLAYBACK_PENDING`
- `DIFFUSERASER_OR_PROTOCOL_ROOT_CAUSE_PENDING`
- `NO_GATE128`
- `NO_OR_DPO`

All 32 existing Gate32 DiffuEraser raw OR candidates were reaudited with
per-sample mp4 generation, sampled frames, object/mask crops, affected-region
crops, and outside crops. The classification remains:

- `medium-hard`: 11
- `trivial-bad`: 21
- `too-close`: 0
- `technical-invalid`: 0

The individual audit found `black_frame_ratio=0.0` for all rows, so the earlier
purple/black impression is partly due to contact-sheet error-map visualization
rather than raw full-video black collapse. The main failure is still real raw
task-region mismatch. Interactive video playback is not available in the
current tool channel, so `VIDEO_REVIEW_PASS`, `LOSER_UTILITY_PASS`, and
`DATA_READY` remain unset.

The DiffuEraser OR root-cause matrix is still pending; no conclusion has been
made about 6-step sampler, mask dilation, PCM, official core, or SFT domain
shift.

Reports:

- `reports/exp25_gate32_individual_video_reaudit.md`
- `reports/exp25_diffueraser_or_root_cause_matrix.md`

## 2026-06-24 Gate32 Dense Temporal Review Fallback

Status: `GATE32_FINAL_DENSE_REVIEW_IMPLEMENTED_PENDING_PAI_RUN`

A fresh readback confirmed that the previous individual Gate32 audit was only a
sampled frame/crop audit and explicitly marked visual review pending. The
reaudit script now supports a denser temporal review fallback for this headless
execution channel:

- 16 uniformly sampled frames;
- start / middle / end;
- mask-area max frame;
- mask-error max frame;
- temporal-difference top-3 frames;
- object/mask, affected-region, outside crops;
- animated GIF fallback;
- blind/informed/final classification fields.

This does not regenerate Gate32, does not expand Gate128, and does not mark
`DATA_READY`. The DiffuEraser OR root-cause matrix remains required before
judging loser utility.

Report:

`reports/exp25_gate32_final_dense_review_readback.md`

## 2026-06-24 Gate32 Final Dense Temporal Review

Status:

- `GATE32_FINAL_DENSE_REVIEW_COMPLETE_YIELD_POOR`
- `DIFFUSERASER_OR_PROTOCOL_ROOT_CAUSE_PENDING`
- `NO_GATE128`
- `NO_OR_DPO`

The existing 32 canonical DiffuEraser raw OR candidates were reaudited with the
dense temporal fallback. Results remain:

- `MEDIUM_HARD_ELIGIBLE`: 11
- `TRIVIAL_BAD`: 21
- `TOO_CLOSE`: 0
- `TECHNICAL_INVALID`: 0

The review confirms that the failures are not whole-video black-frame collapse:
`black_frame_ratio=0.0` for all rows. The dominant issue is task/mask-region
semantic mismatch and local raw loser quality under the current raw6 stack.

Large visual artifacts were generated under the PAI/HAL review artifact roots
but are not committed. This is not `DATA_READY`; the DiffuEraser OR root-cause
matrix remains required before judging loser utility.

Reports:

- `reports/exp25_gate32_final_video_review.md`
- `reports/exp25_gate32_final_video_review.csv`
- `reports/exp25_gate32_final_video_review_summary.json`
## 2026-06-25 PAI Pre-Maintenance Persistence

Status: `BLOCKED_NAS_PERMISSION`

Before launching the Exp25 DiffuEraser OR root-cause matrix, the PAI `/home/hj`
Gate32 dense review artifacts were audited for durable NAS persistence:

- source: `/home/hj/exp25_gate32_dense_review_runs`
- files: `99`
- bytes: `66982608`

The intended NAS target under
`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_vor_or_preference_data/`
could not be created because SSH user `hj` has no write permission to the NAS
project root or the existing `logs/autoresearch` directory. `sudo -n` is not
available and root SSH with the current key failed.

No new Exp25 root-cause matrix or OR candidate generation was launched after
this blocker. Report:

`reports/pai_premaintenance_output_persistence.md`

## 2026-06-25 PAI Post-Maintenance Permission Recovery

Status:

- `PAI_POSTMAINTENANCE_PERMISSIONS_RECOVERED`
- `BLOCKER_RESOLVED`
- `DIFFUSERASER_OR_PROTOCOL_ROOT_CAUSE_READY_TO_RESUME`
- `NO_GATE128`
- `NO_OR_DPO`

The user executed the minimal PAI root permission repair successfully. HAL SSH
as `hj` rechecked the actual PAI state and confirmed:

- DiffuEraser converted weights are readable/executable.
- `brushnet/config.json` is readable.
- `unet_main/config.json` is readable.
- Exp25 experiment output is writable.
- Exp25 autoresearch output is writable.

No ACL installation, apt operation, or broad NAS permission change was repeated
by Codex. The next allowed Exp25 milestone is the post-permission DiffuEraser
asset smoke followed by the fixed 12-sample root-cause matrix.

Reports:

- `reports/exp25_permission_recovery_readback.md`
- `reports/pai_postmaintenance_permission_recovery_final.md`
- `reports/pai_postmaintenance_permission_recovery_final.csv`

## 2026-06-25 Root-Cause Matrix Permission Blocker

Status:

- `GATE32_FINAL_DENSE_REVIEW_COMPLETE_YIELD_POOR`
- `DIFFUSERASER_OR_PROTOCOL_ROOT_CAUSE_BLOCKED_BY_WEIGHT_PERMISSION`
- `NO_GATE128`
- `NO_OR_DPO`

A fixed 12-sample root-cause matrix manifest was built from Gate32 dense-review
evidence, covering medium-hard/trivial-bad rows across REAL and BLENDER sources.

Manifest:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_vor_or_preference_data/root_cause_matrix_20260625/root_cause_sample12_manifest.jsonl`

SHA256:

`d1a7ef848ce1f5777ae80f1655c581fa5328d108fab497693d8afddf750afa49`

The first runnable stack comparison failed before model inference because PAI
user `hj` cannot read the DiffuEraser checkpoint directory:

`/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000`

The failing file access is `brushnet/config.json` / `unet_main/config.json`
with `PermissionError: [Errno 13] Permission denied`. This is an infrastructure
permission blocker, not a DiffuEraser OR-quality result. No Gate128 expansion,
multi-model OR pool construction, or OR-DPO training was launched.

Report:

`reports/exp25_diffueraser_or_root_cause_matrix_20260625_status.md`

## 2026-06-25 PAI Pre-Maintenance Persistence Resolved

Status:

- `PAI_PREMAINTENANCE_PERSISTENCE_PASSED`
- `DIFFUSERASER_OR_PROTOCOL_ROOT_CAUSE_PENDING`
- `NO_GATE128`
- `NO_OR_DPO`

The NAS permission blocker was resolved from the PAI WebIDE root terminal by
granting `hj` write access to the required autoresearch/runtime directories.
HAL SSH as `hj` verified write access.

The existing PAI `/home` artifacts were persisted to NAS:

- Exp25 Gate32 dense review: `99` files, `66982608` bytes, inventory OK,
  SHA256 OK.
- Cross-track Exp26 Gate64 official generation: `14408` files,
  `8405904095` bytes, inventory OK, SHA256 OK.

Markers created:

- `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/EXP25_GATE32_REVIEW_PERSISTED_TO_NAS`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/EXP26_GATE64_PERSISTED_TO_NAS`

Report:

`reports/pai_premaintenance_output_persistence.md`
