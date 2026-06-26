# Exp26 VideoPainter DPO v2 Status

- branch: `research/exp26-videopainter-dpo-v2`
- worktree: `/home/hj/H20_Video_inpainting_DPO_exp26_videopainter`
- status: `VP2_STATIC_FIXES_STARTED`
- copied source: `exp14_adapter_videopainter/`
- current scope: static v2 trainer fixes and unit tests
- GPU work: not started
- DAVIS50: not started

Initial fixes:

- official optimizer fields exposed;
- `noised_image_dropout` wired into image latent preparation;
- first-frame consistency helper added;
- native 49-frame policy enforced, 13F requires plumbing flag;
- formal mode now rejects 16-frame inputs instead of trimming them to 13;
- `--first_frame_gt` / `--no-first_frame_gt` now controls first-frame
  consistency;
- official optimizer/scheduler parser added for locked parity config;
- `itertools.cycle(loader)` removed;
- loser-dominant definition aligned with project diagnostics;
- strict checkpoint reload helper added.
# Exp26 Registry Status

L0_L4_PASSED
FORMAL_49F_SOURCE_BLOCKED

## 49F Source Diagnostic

- Source root: `/mnt/workspace/hj/nas_hj/data/external/ytbv_2019_full_resolution/train`
- Required valid candidates: 640
- Valid candidates: 0
- Failed candidates: 3471
- Max frame count seen: 36
- Max mask count seen: 36
- Gate64 official baseline self-loser generation: not launched.

Reason: active YouTube-VOS source is a sparse extraction and does not satisfy formal 49-frame input requirements.

## VOR-BG Source-Only Fallback

- status: `VOR_BG_SOURCE_SPLIT_LOCKED_PENDING_EXTRACTION_MASKS`
- source: VOR-Train BG clean videos
- train/search/shadow: 128 / 32 / 32
- split isolation: scene-group disjoint
- overlaps: train/search 0, train/shadow 0, search/shadow 0
- manifest hashes:
  - train `68a862c18ad46271757d96c6b5483c76aeb2cfdfb5cae21a452e215b735daaa6`
  - search `41c6571d26e4a5130818dd50fbbe1314c9d953284561a3cd20f630572f7c2a71`
  - shadow `0338dba1513cfe0e5dd85cbf793b3782902b981ac9610b0e472c6a048f738c02`

Gate64 has not started. Selected sources still require selective extraction,
exact 49-frame decode audit, and generated moving BR masks.
## 2026-06-23 Overnight Autonomous Controller

- Status: `OVERNIGHT_WAITING_GPU_OFFICIAL_PROBE_PENDING`.
- Runtime controller on PAI:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260623`.
- CPU mask distribution audit completed and recorded:
  `reports/exp26_br_mask_distribution_audit_fast512.md`.
- Probe4 masks remain plumbing-valid ellipse masks and are not a final Gate16
  protocol.
- Official 49F VideoPainter Probe4 inference is still pending a free allowed
  GPU; no Gate16/Gate64 generation or training has started.

## 2026-06-23 Probe4 Official 49F Inference

- Status: `PROBE4_OFFICIAL_49F_INFERENCE_PASSED`.
- GPU: `2`.
- Rows: `4/4`.
- Frames: `49` for every row.
- Output:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260623/exp26_probe4_official_inference`.
- Summary:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260623/exp26_probe4_official_inference/probe4_official_inference_summary.json`.
- Report: `reports/exp26_probe4_official_inference_result.md`.
- Gate16/Gate64 and DPO training remain unstarted.

## 2026-06-24 Probe4 and Gate16 Review

- Status: `PROBE4_PASSED_GATE16_REVIEW_FAILED`.
- Controller run:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260624_three_lane_retry2`.
- Probe4: `4/4` official 49F inference outputs passed review.
- Gate16 source materialization: `25/32` valid formal 49F candidates; selected `16/16`.
- Gate16 official inference: `16/16`, all `49` frames.
- Gate16 review: `15/16` passed, `1/16` failed.
- Failed sample: `vp2_gate16_BLENDER_CON001_00742`, mask PSNR `1.3500`.
- Gate64: not launched.
- Report: `reports/exp26_probe4_gate16_review_20260624.md`.

## 2026-06-24 Gate16 Reclassification

- Status:
  - `PROBE4_PASSED`
  - `GATE16_15_OF_16`
  - `GATE16_METRIC_GATE_PASS_VISUAL_REVIEW_PENDING`
  - `NO_GATE64`
  - `NO_DPO_TRAINING`
- Existing Gate16 rows reclassified: `16/16`.
- Classification: `medium-hard=15`, `trivial-bad=1`, `technical-invalid=0`.
- Failed row retained as true model failure: `vp2_gate16_BLENDER_CON001_00742`.
- Original numeric gate criteria pass, but interactive video playback is still pending, so `GATE16_PASSED_WITH_REJECTION` is not set.
- Report: `reports/exp26_gate16_reclassification.md`.

## 2026-06-24 Gate16 Final Video Review

- Status: `GATE16_PASSED_WITH_REJECTION`.
- Existing Gate16 outputs reviewed: `16/16`.
- Review method: opened every per-sample contact sheet and every generated
  dense16 temporal review pack.
- Final buckets: `medium-hard=10`, `hard-plausible=5`,
  `trivial-bad=1`, `technical-invalid=0`.
- Failed row retained as true model failure:
  `vp2_gate16_BLENDER_CON001_00742`.
- Gate64 may be prepared as the next milestone after a fresh readback.
- No Gate64, DPO micro-training, or long training has been launched by this
  milestone.
- Report: `reports/exp26_gate16_final_video_review.md`.

## 2026-06-24 Gate64 Mixed-Mask Protocol

- Status: `GATE64_PROTOCOL_LOCKED_PENDING_PAI_GENERATION`.
- Config: `exp26_videopainter_dpo_v2/configs/vp2_mixed_br_mask_v1.json`.
- Manifest: `exp26_videopainter_dpo_v2/manifests/vp2_gate64_source_manifest.jsonl`.
- Manifest SHA256: `b904be82d58ab7cd897c6759b7351e262f61397d9f90d84df05ae42300dbffb6`.
- Rows / scene groups: `64 / 64`.
- Distribution: irregular free-form 16, object-like polygon 16, soft blob 8,
  edge-touch free-form 8, ellipse/circle subset 8, thin-structure free-form 8.
- Gate64 generation: not launched.
- VideoPainter DPO: not launched.
- Current blocker for PAI execution: SSH host-key verification changed for
  `47.103.26.60`, presented ED25519 fingerprint
  `SHA256:xDOCAS/+fw0Bs5m9HizeRi1mkYOcIotlm4CxcfWwpqk`.

## 2026-06-24 Gate64 Generation Launcher

Status: `GATE64_GENERATION_IMPLEMENTED_PENDING_PAI_RUN`

- Gate64 generation readback completed.
- Mixed-mask protocol implementation bug fixed before generation.
- Added exact VOR-Train/BG selective extractor.
- Added official Gate64 generation runner and PAI launcher.
- Validation passed: py_compile, 23 unit tests, bash -n, git diff --check.
- Gate64 generation pending PAI snapshot run.
- VideoPainter DPO not started.

## 2026-06-25 Gate64 Official Generation

Status: `GATE64_GENERATION_PARTIAL_SOURCE_PASS_56_OF_64`

- PAI run root:
  `/home/hj/exp26_gate64_runs/gate64_official_43597cf_20260625_031155`
- Code snapshot commit: `43597cf66c106ceddcdb384ec7207993662d3f1e`
- Extraction: `64/64 OK`
- Formal 49F materialization: `56/64 OK`, `8/64 duplicate-frame source failures`
- Mask generation: `56/56 OK`
- Official VideoPainter generation: `56/56 OK`
- Output frames/videos/contact sheets: `56`
- GPU after completion: all cards 0 MiB
- Report: `reports/exp26_gate64_generation_status_20260625.md`

Gate64 evidence review is pending. This is not yet `DATA_READY`, and no
VideoPainter DPO training has started.

## 2026-06-25 PAI Pre-Maintenance Persistence

Status: `BLOCKED_NAS_PERMISSION`

- PAI `/home` source: `/home/hj/exp26_gate64_runs/gate64_official_43597cf_20260625_031155`
- files / bytes: `14408 / 8405904095`
- intended NAS target:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/gate64_official_43597cf_20260625_031155`
- blocker: SSH user `hj` cannot write under the NAS project root.
- markers not created: `EXP26_GATE64_PERSISTED_TO_NAS`,
  `EXP25_GATE32_REVIEW_PERSISTED_TO_NAS`
- report: `reports/pai_premaintenance_output_persistence.md`

No new Exp26 GPU work was started after this blocker.

## 2026-06-25 PAI Pre-Maintenance Persistence Resolved

Status:

- `PAI_PREMAINTENANCE_PERSISTENCE_PASSED`
- `GATE64_VISUAL_REVIEW_PENDING`
- `NO_VIDEOPAINTER_DPO`

Persisted artifacts:

- Exp26 Gate64 official generation:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/gate64_official_43597cf_20260625_031155`
  with `14408` files / `8405904095` bytes / inventory OK / SHA256 OK.
- Exp25 Gate32 dense review:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_vor_or_preference_data/gate32_dense_review_20260625`
  with `99` files / `66982608` bytes / inventory OK / SHA256 OK.

Completion markers are present under:

`/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/`

## 2026-06-25 Gate64 Duplicate Source Audit

Status:

- `DUPLICATE_SOURCE_AUDIT_COMPLETE`
- `GATE64_SOURCE_VALID_ROWS_56_OF_64`
- `GATE64_VISUAL_REVIEW_PENDING`
- `NO_VIDEOPAINTER_DPO`

The 8 Gate64 materialization failures were audited after persistence. All 8
are real pixel-duplicate source clips in the selected first 49 frames under
sequential OpenCV decode. They are not random-seek artifacts and not
VideoPainter inference failures.

Reports:

- `reports/exp26_gate64_duplicate_source_audit.md`
- `reports/exp26_gate64_duplicate_source_audit.csv`
- `reports/exp26_gate64_duplicate_source_audit.json`

## 2026-06-25 Gate64 Human Visual Review

Status:

- `GATE64_EVIDENCE_REVIEW_COMPLETE_MP4_PLAYBACK_PENDING_POOL_NOT_DATA_READY`
- `NO_VIDEOPAINTER_DPO`

Reviewed generated outputs: `56/56`.

Classification:

- `medium-hard=31`
- `hard-plausible=16`
- `too-close=1`
- `trivial-bad=8`

Decision:

- eligible after evidence review: `47`
- rejected too-close: `1`
- rejected trivial/technical: `8`

VideoPainter DPO remains unstarted. Next allowed step is balanced manifest
construction from the 47 eligible rows, not training.

Reports:

- `reports/exp26_gate64_human_visual_review.md`
- `reports/exp26_gate64_human_visual_review.csv`
- `reports/exp26_gate64_human_visual_review_summary.json`

## 2026-06-25 Post-Maintenance Gate64 Source Repair

Status:

- `GATE64_SOURCE_REPAIR_COMPLETE`
- `GATE64_FORMAL_VALID_64_OF_64`
- `GATE64_EVIDENCE_REVIEW_COMPLETE_MP4_PLAYBACK_PENDING`
- `GATE64_PRIMARY32_DRAFT_LOCKED_MP4_PLAYBACK_PENDING`
- `NO_VIDEOPAINTER_DPO`

Deep duplicate-source audit reclassified the 8 formal-source failures as
static pixel duplicates with valid 49-frame index/timestamp evidence. The
materializer now allows these static duplicates and records duplicate groups in
diagnostics.

Repair generation and review:

- materialization: `8/8`
- masks: `8/8`
- official VideoPainter outputs: `8/8`, `49` frames each
- repair visual buckets: `6` medium-hard, `2` hard-plausible

Combined Gate64:

- formal-valid: `64/64`
- evidence-reviewed outputs: `64/64`; strict mp4 playback pending
- eligible after evidence review: `55`
- rejected: `9`
- primary-32 draft manifest locked:
  `exp26_videopainter_dpo_v2/manifests/vp2_gate64_primary32_visual_reviewed_comp.jsonl`

DPO remains unstarted because the Exp26 NAS experiment output directory is not
writable by PAI user `hj`.

Reports:

- `reports/exp26_gate64_source_repair_readback.md`
- `reports/exp26_gate64_duplicate_source_deep_audit.md`
- `reports/exp26_gate64_repair_human_visual_review.md`
- `reports/exp26_gate64_final_readiness.md`

## 2026-06-25 PAI Post-Maintenance Permission Recovery

Status:

- `PAI_POSTMAINTENANCE_PERMISSIONS_RECOVERED`
- `BLOCKER_RESOLVED`
- `GATE64_FINAL_TEMPORAL_REVIEW_READY_TO_RUN`
- `NO_VIDEOPAINTER_DPO_YET`

Confirmed on PAI host `dsw-753014-85f54df947-bkp7h` as user `hj`:

- Exp26 NAS experiment output: writable.
- Exp26 NAS autoresearch output: writable.
- Gate64 output root: readable/writable.

Reports:

- `reports/exp26_permission_recovery_readback.md`
- `reports/pai_postmaintenance_permission_recovery_final.md`
- `reports/pai_postmaintenance_permission_recovery_final.csv`

## 2026-06-25 Gate64 Final Temporal Review

Status:

- `GATE64_DATA_READY`
- `VIDEO_REVIEW_PASS`
- `PRIMARY32_FINAL_LOCKED`
- `NO_VIDEOPAINTER_DPO_YET`

Final Gate64 buckets:

- medium-hard: `37`
- hard-plausible: `18`
- too-close: `1`
- trivial-bad: `8`
- technical-invalid: `0`
- eligible: `55`

Primary-32:

- manifest:
  `exp26_videopainter_dpo_v2/manifests/vp2_gate64_primary32_final.jsonl`
- SHA256:
  `82f661f2f30a581a213972533817624217eabb97eba7aaeedc00ee2109e4e716`
- class balance: `16` medium-hard + `16` hard-plausible
- search-dev scene overlap: `0`
- shadow-dev scene overlap: `0`
- path/frame validation: `32/32` passed, all winner/comp/raw/mp4 decode as `49` frames

Training loser semantics:

- primary training loser: hard-composited VideoPainter output through
  `final_loser_video_path`
- raw output: diagnostic and ablation only

Reports:

- `reports/exp26_gate64_final_temporal_review.md`
- `reports/exp26_gate64_primary32_final.md`
- `reports/exp26_gate64_primary32_path_frame_validation.csv`

## 2026-06-25 Primary-32 L0/L1

Status:

- `VP_L0_L1_PASSED`
- `TECHNICAL_PASS`
- `NO_10STEP_YET`

Results:

- L0 DPO loss: `0.6931471824645996`
- L0 policy grad norm: `14.379858399808493`
- L0 reference gradients: `false`
- L1 policy delta norm: `1.6732795822542237`
- L1 reference delta norm: `0.0`
- strict reload max abs diff: `0.0`
- checkpoint:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp26_videopainter_dpo_v2/vp_l0_l1_primary32_retry3_20260625_125902/one_step_run/checkpoint-1`

Reports:

- `reports/exp26_vp_l0_l1.md`
- `reports/exp26_vp_l0_l1.json`
- `reports/exp26_vp_l0_l1_diagnostics.csv`

## 2026-06-25 Search-Dev Step0 Baseline

Status:

- `VP_STEP0_BASELINE_LOCKED`
- `TEMPORAL_REVIEW_PASS`
- `10STEP_READY`
- `NO_10STEP_RESULT_YET`

Search-dev rows: `32`.

Temporal evidence reviewed: `32 / 32`; reviewer pass: `32 / 32`.

Comp step0 metrics:

- PSNR: `24.301897366442233`
- SSIM: `0.871557953992803`
- LPIPS: `0.07080062118242197`
- Ewarp: `8.04273951919754`
- strict mask PSNR: `16.012427341260924`
- boundary PSNR: `16.01195236353609`

Raw output remains diagnostic only; comp output is the primary loser protocol.

Reports:

- `reports/exp26_vp_step0_baseline.md`
- `reports/exp26_vp_step0_baseline.csv`
- `reports/exp26_vp_step0_visual_review.csv`

## 2026-06-25 Primary-32 10-Step Gate

Status:

- `VIDEOPAINTER_10STEP_GATE_PASSED`
- `TRAINING_PASS`
- `DENSE_EVIDENCE_REVIEW_PASS_FOR_50STEP_GATE`

Final primary-32 10-step DPO completed on PAI. Checkpoint-1 and checkpoint-10
strict reload/preflight passed. Dense temporal evidence and crop sheets were
reviewed for all `32/32` search-dev rows.

Step10 comp vs step0 comp:

- PSNR `+0.977252`
- SSIM `+0.032641`
- LPIPS `-0.004499`
- Ewarp `-1.301457`
- strict mask PSNR `+0.975192`
- boundary PSNR `+5.082206`

No global collapse, frame-order mismatch, or systematic new artifact was found
in dense evidence. This authorizes only the conditional 50-step gate.

## 2026-06-25 Primary-32 50-Step Gate

Status:

- `VIDEOPAINTER_ADAPTER_POSITIVE`
- `TRAINING_PASS`
- `SEARCH_DEV_MICRO_GATE_ONLY`
- `NOT_SCIENTIFIC_POSITIVE`
- `NO_RCFPO`
- `NO_100STEP_OR_LONGER`

Final primary-32 50-step DPO completed on PAI. Checkpoint-10/20/30/40/50
strict reload/preflight passed. Dense temporal evidence and crop pages were
reviewed for all `32/32` search-dev rows.

Step50 comp vs step0 comp:

- PSNR `+4.816168`
- SSIM `+0.087883`
- LPIPS `-0.044059`
- Ewarp `-7.055122`
- strict mask PSNR `+4.942246`
- boundary PSNR `+12.111889`

Paired PSNR win rate was `0.718750`; bootstrap 95% CI was
`[+2.648960, +7.234666]` with probability(delta > 0) `1.000000`.

No global collapse, frame-order mismatch, first-frame failure, systematic
outside damage, or gate-blocking flicker/ghosting was found. Remaining visible
failures are local mask/affected-region artifacts. This result is a
search-dev micro-training gate, not a final benchmark or scientific-positive
claim.

## 2026-06-25 Shadow-Dev Confirmatory Validation

Status:

- `SHADOWDEV_CONFIRMATORY_PROTOCOL_AUDITED`
- `SHADOWDEV_INTEGRITY_PASS_PENDING_MATERIALIZATION`
- `CHECKPOINT_IDENTITY_PENDING_PAI_PATHS`
- `NO_RETRAINING`
- `NO_100STEP_OR_LONGER`
- `NO_RCFPO`

Primary comparison is fixed Step50 versus fixed Step0 on the locked independent
shadow-dev split. Step10 and Step30 are trajectory diagnostics only and cannot
replace Step50 after seeing shadow-dev results.

The left CLI Exp25/Exp27/Exp28 controller remains protected and read-only from
this right-side Exp26 session. GPU1-4 are reserved for that controller; Exp26
shadow-dev may only use dynamically eligible GPU0/5/6/7.

Reports:

- `reports/exp26_vp_shadowdev_readback.md`
- `reports/exp26_vp_shadowdev_integrity_audit.md`
- `reports/exp26_vp_shadowdev_checkpoint_identity.md`

## 2026-06-26 Shadow-Dev Confirmatory Validation

Status:

- `VIDEOPAINTER_SHADOWDEV_CONFIRMED`
- `CROSS_BACKBONE_ADAPTER_EVIDENCE_CONFIRMED`
- `NO_100STEP_OR_LONGER`
- `NO_RCFPO`

Fixed Step50 from `vp_primary32_50step_20260625_171032` was compared against
fixed Step0 on the locked 32-row shadow-dev split. Step10/Step30 remain
trajectory diagnostics only.

Primary frame1-48 comp deltas:

- strict mask PSNR `+5.186942`, win rate `0.781250`,
  probability improved `1.000000`
- boundary PSNR `+12.175098`, win rate `1.000000`,
  probability improved `1.000000`
- LPIPS `-0.040142`, win rate `0.937500`
- Ewarp `-8.378847`, win rate `0.968750`
- whole comp PSNR `+5.160739`

No unexpected winner leakage was detected. Visual review covered `32/32`
shadow rows: Step50 better `25`, tie `3`, Step0 better/new Step50 artifact
`4`. Seed robustness passed on a fixed 16-row subset across seeds
`20260619`, `20260620`, and `20260621`.

This confirms held-out VideoPainter evidence for the current VOR-BG
distribution, but it is not a universal-adapter or final SOTA claim.

## 2026-06-26 Post-Confirmation Sanity Audit

- status: `EXP26_POSTCONFIRMATION_SANITY_AUDIT_PASSED`
- scope: read-only audit of the completed fixed Step50 vs fixed Step0
  shadow-dev confirmation.
- evidence:
  - fixed trajectory: `vp_primary32_50step_20260625_171032`
  - primary32/search/shadow SHA256 identities unchanged
  - train/search/shadow overlap `0 / 0 / 0`
  - no unexpected GT/winner leakage in `128` audited rows
  - shadow visual review `32 / 32`
  - seed robustness pass `3 / 3`
- reports:
  - `reports/exp26_postconfirmation_readback.md`
  - `reports/exp26_postconfirmation_sanity_audit.md`
  - `reports/exp26_postconfirmation_sanity_audit.csv`
  - `reports/exp26_postconfirmation_sanity_audit.json`

## 2026-06-26 External 49F Inventory

- status: `EXP26_EXTERNAL_49F_INVENTORY_COMPLETE`
- candidate directories: `2024`
- valid clean 49F sources: `54`
- selected rows: `32`
- selected manifest:
  `exp26_videopainter_dpo_v2/manifests/vp2_external_49f_validation_16_or_32.jsonl`
- selected SHA256:
  `be118a7ce7d462bda6c339053d0c112994c8da7fab6cf00a4ee5dae87b628e5a`
- source family: DAVIS-derived `gt_frames`; `comparison.mp4` files are rejected
  as clean sources.
- no masks, prompts, seeds, inference outputs, or checkpoint choices were
  generated from these rows yet.

## 2026-06-26 External Validation Preregistration

- status: `EXP26_EXTERNAL_VALIDATION_PREREGISTERED`
- rows: `32`
- source manifest SHA256:
  `be118a7ce7d462bda6c339053d0c112994c8da7fab6cf00a4ee5dae87b628e5a`
- preregistered manifest:
  `exp26_videopainter_dpo_v2/manifests/vp2_external_validation_preregistered.jsonl`
- preregistered manifest SHA256:
  `69ecd96d4b25da702229df2d45bf1343ad5e7ef5385cbd32d24ce61644e4bc2c`
- mask manifest:
  `exp26_videopainter_dpo_v2/manifests/vp2_external_validation_masks.jsonl`
- mask manifest SHA256:
  `f646792469f53a8122fe341be5988344ba7b32d33b3a53593d558e227aed138b`
- output root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/postconfirmation_20260626/external_validation/preregistered`
- protocol: fixed Step0-vs-Step50, frame1-48 primary metrics, first-frame GT,
  seed `20260619`, mask seed `20260623`, `720x480`, 20 inference steps,
  guidance `6.0`, bf16.
- no external inference, metric, or visual review has been run yet.

## 2026-06-26 External Validation Generation

- status: `EXP26_EXTERNAL_GENERATION_COMPLETE`
- leakage: `NO_UNEXPECTED_WINNER_LEAKAGE_DETECTED`
- generated checkpoints: `Step0`, `Step10`, `Step30`, `Step50`
- rows per checkpoint: `32 / 32`
- leakage rows audited: `128`
- leakage flagged rows: `0`
- output root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/postconfirmation_20260626/external_validation/`
- reports:
  - `reports/exp26_external_validation_generation_status.md`
  - `reports/exp26_external_validation_generation_status.csv`
  - `reports/exp26_external_validation_leakage_audit.md`
  - `reports/exp26_external_validation_leakage_audit.csv`

No external checkpoint reselection, retraining, mask reselection, seed
reselection, or source-row replacement was performed.
