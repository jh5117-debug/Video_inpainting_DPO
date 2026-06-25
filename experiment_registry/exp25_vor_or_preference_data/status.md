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
- status: CORE_DOWNLOAD_COMPLETE
- completed time: 2026-06-23T00:08:45+0200
- completed files: 37 / 37
- completed bytes: 363730944386 / 363730944386
- PAI final files: 37
- PAI partial files: 0
- PAI bad files: 0
- HAL staging final size: 1.0K

## Safety

This track is download-only. It does not enter Exp23 worktrees, use GPUs, run inference, generate losers, or start DPO training. Tokens remain only under `/home/hj/.cache/huggingface_effecterase_auth` and are not copied to PAI or committed.

## Completion

Core compressed files are present under the fixed dataset revision directory on PAI:

`/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/downloads/fa09dc61128ca0418a4a13364d97a08018ea9cc7`

The PAI completion marker exists at:

`/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/runtime/CORE_DOWNLOAD_COMPLETE`

The transfer manifest contains 37 VERIFIED rows and zero HAL/PAI SHA256 mismatches. Final independent PAI inventory verification completed with `ok=true`: 37 OK files, 0 partial files, 0 bad files, 363730944386 final bytes, and contiguous VOR-Train archive parts 000-031.

## Current Work

Status: `SELECTIVE_OR_DATA_TOOLING_STARTED`

Added initial Exp25-only scripts for lightweight archive inspection, resumable
member indexing, safe selective extraction, extracted-subset validation, and
canonical VOR OR manifest semantics. No VOR archive was decompressed in full,
no loser generation was started, and no GPU training was launched in this step.

Archive audit: `reports/vor_archive_integrity.md` passed on PAI with matching
part counts and byte counts for VOR-Eval, VOR-Train-MASK, and VOR-Train. Stream
probe opened all three split gzip/tar streams and found zero unsafe paths in
the probed members.

VOR-Eval extraction: completed on PAI under
`/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/extracted/vor_eval_full`.
`BG`, `FG_BG`, and `MASK` each contain 43 aligned mp4 files and are reserved for
held-out final evaluation only.
# Exp25 Registry Status

CORE_DOWNLOAD_COMPLETE
FULL_MEMBER_INDEX_COMPLETE
TRIPLET_PAIRING_CONFIRMED
GROUP_SPLITS_LOCKED
GATE128_EXTRACTED
THREE_MODEL_SMOKE_PARTIAL_BLOCKED

## Current Smoke Status

- ProPainter: PASS, 6/6.
- DiffuEraser: BLOCKED by PCM LoRA adapter API mismatch.
- EffectErase: BLOCKED, verified wrapper/checkpoint unresolved.

Full Gate128 OR loser generation has not started.

## 2026-06-23 Stack Audit

- DiffuEraser core checkpoint is full `brushnet/` + `unet_main`, not a LoRA training adapter.
- The current blocker is PCM inference acceleration compatibility in the OR path.
- `DE_OFFICIAL_PCM2` is blocked pending official-pinned environment smoke.
- `DE_CANONICAL_NO_PCM` is BR-verified but OR/VOR-pending.
- EffectErase official repo is cloned on HAL, but `EffectErase.ckpt` and `Wan2.1-Fun-1.3B-InP` are missing.

## 2026-06-23 Stack Audit v2

- Added Exp25-only DiffuEraser OR wrapper with explicit `--pcm_mode official_pcm2|none`.
- `official_pcm2` preserves the official PCM LoRA acceleration path and remains an accelerated-policy variant.
- `none` removes the PCM load only inside a temporary overlay copy, sets an explicit no-PCM scheduler step/guidance identity, and is the strict on-policy primary candidate pending VOR smoke.
- ProPainter prior remains a separate `prior_mode=propainter` variable and is retained for the main comparison.
- Gate128 remains blocked until fixed Smoke6 generation, metrics, and visual review pass.

## 2026-06-23 DiffuEraser no-PCM Smoke6

- Generator: `diffueraser_or_none_propainter_62d00ca9c76a`.
- Technical generation: `6/6` fixed Smoke6 samples, 24 frames each, raw/no-comp.
- PAI environment fix: installed public `av==17.1.0` for torchvision video I/O.
- Visual review completed on six contact sheets.
- Decision: `DIFFUERASER_NO_PCM_TECHNICAL_PASS`.
- Gate128 readiness: `false`.
- Reason: insufficient medium-hard loser utility in Smoke6; too many outputs are too close to winner, plus one visible residual-artifact case.

## 2026-06-23 Overnight Autonomous Controller

- Status: `OVERNIGHT_CONTROLLER_RUNNING_WAITING_GPU`.
- Controller PID on PAI: `1903925`.
- Runtime root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260623`.
- PAI runtime code was created from tracked git archives under
  `/mnt/workspace/hj/nas_hj/runtime_code_snapshots/`.
- Fixed six-sample Gate128 materialization for canonical d0 Smoke6 completed:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/materialized/gate128_smoke6_canonical_d0_24f/smoke6_materialized.jsonl`.
- Fresh canonical DiffuEraser Smoke6 d0 is queued and has not run because
  GPU2/4/5/6 are all occupied by other jobs; GPU7 remains excluded by policy.
- EffectErase remote inventory completed on HAL at revision
  `fa09dc61128ca0418a4a13364d97a08018ea9cc7` with `37` required files and
  `363730944386` total bytes.
- Independent PAI EffectErase checksum verification is still running from HAL
  as background PID `3577126`.
- Gate32/Gate128 OR generation remains blocked until the canonical d0 Smoke6
  finishes and receives real metric plus visual review.

## 2026-06-23 Overnight Smoke6 Monitor Update

- Fresh canonical d0 DiffuEraser Smoke6 completed on PAI: `6/6`, `24` frames each.
- Generator id: `diffueraser_or_none_propainter_abd3ad48f60f`.
- Confirmed `pcm_mode=none`, `prior_mode=propainter`, `mask_dilation_iter=0`, `hard_comp=false`.
- Visual review completed on six contact sheets.
- Decision: `TECHNICAL_PASS_QUALITY_YIELD_WEAK`.
- Most samples were too close to winner; one sample showed clear ghosting and useful hard-negative behavior.
- Gate32 remains pending as yield calibration only.
- Report: `reports/exp25_overnight_smoke6_monitor.md`.

## 2026-06-23 Overnight GPU2 Gate32 Completion

- Status: `GATE32_CANONICAL_DIFFUERASER_COMPLETED`.
- Controller fix commit: `22f7c54f49c46655700d7cb4193ac18dfa3bf037`.
- PAI controller PID after fix: `2041604`.
- Runtime root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260623`.
- Gate32 materialization: `32/32`.
- Gate32 DiffuEraser canonical OR generation: `32/32` OK, `24` frames each.
- Generator ID: `diffueraser_or_none_propainter_abd3ad48f60f`.
- Confirmed `pcm_mode=none`, `prior_mode=propainter`, `hard_comp=false`.
- Output root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/preference_candidates/gate32_canonical_raw6_d0`.
- GPU2 released after completion; GPU7 stale allocation was not used.
- Report: `reports/exp25_overnight_gpu2_gate32_completion.md`.

## 2026-06-24 Three-Lane Gate32 Yield Review

- Status: `GATE32_YIELD_REVIEW_COMPLETED`.
- Controller run:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260624_three_lane_retry2`.
- Gate32 candidates reviewed: `32/32`.
- Yield buckets: `medium-hard=11`, `too-close=0`, `trivial-bad=21`.
- Seed2 supplementation: not launched because `too-close=0`.
- Gate128 expansion: not launched.
- Report: `reports/exp25_gate32_yield_review_20260624.md`.

## 2026-06-24 Individual Gate32 Reaudit

- Status:
  - `GATE32_GENERATION_PASSED`
  - `GATE32_QUALITY_YIELD_POOR`
  - `INDIVIDUAL_VIDEO_REAUDIT_FRAME_SAMPLE_COMPLETE_PLAYBACK_PENDING`
  - `DIFFUSERASER_OR_PROTOCOL_ROOT_CAUSE_PENDING`
  - `NO_GATE128`
  - `NO_OR_DPO`
- Existing Gate32 rows audited: `32/32`.
- Frame/crop evidence generated: `32/32`.
- Classification: `medium-hard=11`, `trivial-bad=21`, `too-close=0`, `technical-invalid=0`.
- `black_frame_ratio=0.0` for all rows; the dominant failure is raw task-region mismatch, not whole-video black-frame collapse.
- Interactive mp4 playback is still pending, so no `VIDEO_REVIEW_PASS` or `DATA_READY` status is set.
- Reports:
  - `reports/exp25_gate32_individual_video_reaudit.md`
  - `reports/exp25_diffueraser_or_root_cause_matrix.md`

## 2026-06-24 Gate32 Dense Review Tooling

Status: `GATE32_FINAL_DENSE_REVIEW_IMPLEMENTED_PENDING_PAI_RUN`

- Fresh readback completed.
- Existing Gate32 outputs will be reviewed with denser temporal evidence.
- No Gate32 regeneration, no Gate128, no OR-DPO.
- Root-cause matrix remains pending.

## 2026-06-24 Gate32 Final Dense Temporal Review

Status:

- `GATE32_FINAL_DENSE_REVIEW_COMPLETE_YIELD_POOR`
- `DIFFUSERASER_OR_PROTOCOL_ROOT_CAUSE_PENDING`
- `NO_GATE128`
- `NO_OR_DPO`

Results:

- `MEDIUM_HARD_ELIGIBLE=11`
- `TRIVIAL_BAD=21`
- `TOO_CLOSE=0`
- `TECHNICAL_INVALID=0`

Reports:

- `reports/exp25_gate32_final_video_review.md`
- `reports/exp25_gate32_final_video_review.csv`
- `reports/exp25_gate32_final_video_review_summary.json`
## 2026-06-25 PAI Pre-Maintenance Persistence

Status: `BLOCKED_NAS_PERMISSION`

- PAI `/home` source: `/home/hj/exp25_gate32_dense_review_runs`
- files / bytes: `99 / 66982608`
- intended NAS target:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_vor_or_preference_data/gate32_dense_review_20260625`
- blocker: SSH user `hj` cannot write under the NAS project root.
- markers not created: `EXP25_GATE32_REVIEW_PERSISTED_TO_NAS`,
  `EXP26_GATE64_PERSISTED_TO_NAS`
- report: `reports/pai_premaintenance_output_persistence.md`

No new Exp25 root-cause matrix was started after this blocker.

## 2026-06-25 PAI Pre-Maintenance Persistence Resolved

Status:

- `PAI_PREMAINTENANCE_PERSISTENCE_PASSED`
- `DIFFUSERASER_OR_PROTOCOL_ROOT_CAUSE_PENDING`
- `NO_GATE128`
- `NO_OR_DPO`

Persisted artifacts:

- Exp25 Gate32 dense review:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_vor_or_preference_data/gate32_dense_review_20260625`
  with `99` files / `66982608` bytes / inventory OK / SHA256 OK.
- Exp26 Gate64 official generation:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/gate64_official_43597cf_20260625_031155`
  with `14408` files / `8405904095` bytes / inventory OK / SHA256 OK.

Completion markers are present under:

`/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/`

## 2026-06-25 DiffuEraser OR Root-Cause Matrix Attempt

Status:

- `DIFFUSERASER_OR_PROTOCOL_ROOT_CAUSE_BLOCKED_BY_WEIGHT_PERMISSION`
- `NO_GATE128`
- `NO_OR_DPO`

Root-cause manifest:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_vor_or_preference_data/root_cause_matrix_20260625/root_cause_sample12_manifest.jsonl`

SHA256: `d1a7ef848ce1f5777ae80f1655c581fa5328d108fab497693d8afddf750afa49`

DE-B / DE-C stack comparison could not start model inference because `hj` lacks
read permission on the DiffuEraser checkpoint directory:

`/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000`

This is a PAI asset permission blocker, not a model-quality conclusion.

Report:

`reports/exp25_diffueraser_or_root_cause_matrix_20260625_status.md`

## 2026-06-25 PAI Post-Maintenance Permission Recovery

Status:

- `PAI_POSTMAINTENANCE_PERMISSIONS_RECOVERED`
- `BLOCKER_RESOLVED`
- `DIFFUSERASER_OR_PROTOCOL_ROOT_CAUSE_READY_TO_RESUME`
- `NO_GATE128`
- `NO_OR_DPO`

Confirmed on PAI host `dsw-753014-85f54df947-bkp7h` as user `hj`:

- DiffuEraser converted weights: readable/executable.
- Exp25 NAS experiment output: writable.
- Exp25 NAS autoresearch output: writable.

Reports:

- `reports/exp25_permission_recovery_readback.md`
- `reports/pai_postmaintenance_permission_recovery_final.md`
- `reports/pai_postmaintenance_permission_recovery_final.csv`
