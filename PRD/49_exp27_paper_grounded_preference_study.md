# PRD 49: Exp27 Paper-Grounded Preference Study

## Goal
Build a paper-grounded research track that compares LocalDPO, Diffusion-SDPO, and Linear-DPO against the current LoVI-style video inpainting DPO pipeline, then selects a primary and fallback method only after independent review and exact reproduction gates.

## Guardrails
- Do not modify Exp1-Exp26, shared trainers, or inference/metrics.py.
- Do not start long training before method decision and micro gates.
- Do not claim novelty that is already covered by LocalDPO, Diffusion-SDPO, or Linear-DPO.
- VOR-Eval is final-only and forbidden for selection/threshold/checkpoint choice.

## Current State

Status: `PAPER_REVIEW_COMPLETE`

Status: `EXACT_BASELINE_REPRODUCTION_IN_PROGRESS`

Status: `NO_LONG_TRAINING`

Branch: `research/exp27-paper-grounded-preference-study`.

## 2026-06-23 Update

- Downloaded and hashed LocalDPO, Diffusion-SDPO, and Linear-DPO PDFs.
- Cloned official repositories and recorded commits/licenses.
- Completed five independent review passes A-E.
- Completed CPU parity helpers:
  - SDPO scalar safe-lambda parity passed.
  - Linear-DPO utility and EMA parity passed.
  - LocalDPO official random-mask code is blocked by a runtime error in the official code path.
- Selected primary candidate:
  `RC-FPO`, Restoration-Critical Failure-Structured Preference Optimization.
- Selected fallback:
  `ST-Pref`, Stage-Aware Spatial/Temporal Preference Decomposition.
- Paused Region-SDPO and Linear-DPO as baselines/ablations rather than primary novelty.

No long Exp27 training has been started.

## 2026-06-23 LocalDPO Runtime Diagnosis

The official LocalDPO random mask generator fails in the current dependency
environment even under the official default image size. Root cause: official
`random_mask_gen.py` reads a 4-channel `tostring_argb()` matplotlib canvas
buffer and reshapes it as 3-channel RGB. After that is fixed, the file also
uses `cv2.resize` without importing `cv2`.

Exp27 now adds an isolated compatibility wrapper that does not edit the
official clone:

`exp27_paper_grounded_preference_study/code/localdpo_compat.py`

Mask-only compatibility status:

`OFFICIAL_CODE_COMPATIBILITY_PATCH_MASK_ONLY_PASSED`

Faithful LocalDPO baseline remains incomplete until progressive corruption,
outside latent reinjection/fusion, six-video pair smoke, and 1/10-step training
smoke pass.

SDPO and Linear-DPO remain toy parity only; real DiffuEraser-batch parity is
pending.

## 2026-06-23 LocalDPO Fusion Primitive

Added:

`exp27_paper_grounded_preference_study/code/localdpo_full_adapter.py`

This isolates LocalDPO's core outside-preservation semantics for the
DiffuEraser adaptation:

- task mask, corruption mask, and restoration-critical region are distinct;
- corruption-mask inside uses the denoised/current latent;
- outside the corruption mask reinjects the re-noised original latent at every
  progressive denoising step.

This is an algorithm-primitive parity step, not a full LocalDPO baseline. The
remaining required gates are single-video local corruption, six-video pair
generation, real DiffuEraser-batch SDPO parity, real DiffuEraser-batch
Linear-DPO parity, and 1/10-step micro training.

## 2026-06-23 PAI Official Cache Sync and CPU Primitive Parity

Pinned official code caches were synced from HAL to PAI:

- Local-DPO `7528e966b17283cfa638577827e456737335f030`
- Diffusion-SDPO `84fb241c1b89705a247da8b0d6047798ca49830d`
- Linear-DPO `663179c7adbbbd2d77b97b5841534447eb291ebd`

PAI cache root:

`/mnt/nas/hj/video_dpo_paper_code_cache/`

The official repos remain read-only and are not committed.

Exp27 now installs a narrow runtime compatibility shim for the pinned
Local-DPO code: modern Matplotlib returns ARGB bytes and the official file
expects RGB bytes; the official file also references `cv2` without importing
it. The shim lives only in Exp27 adapter code and does not modify the official
cache.

PAI CPU primitive parity result:

`PASSED`

- LocalDPO official mask generation passed deterministically.
- LocalDPO latent fusion / outside reinjection passed.
- Diffusion-SDPO lambda extraction passed with `max_abs_diff=0.0`.
- Linear-DPO primitive and EMA update passed with `ema_max_abs_diff=0.0`.

Reports:

- `reports/exp27_official_cache_sync.md`
- `reports/exp27_official_cache_manifest.csv`
- `reports/exp27_cpu_primitive_parity_after_cache_sync.md`

Still pending:

- real DiffuEraser batch parity for SDPO;
- real DiffuEraser batch parity for Linear-DPO Frozen and EMA;
- faithful LocalDPO data and original objective baseline;
- all 1/10/50 studies and RC-FPO.

## 2026-06-23 GPU2 Real-Batch Parity

The overnight controller used GPU2 to run Exp27 real-batch objective plumbing
checks after Exp26 Probe4 inference completed.

Results:

- SDPO real-batch parity: `passed`.
- SDPO objective: `0.010575979948043823`.
- SDPO grad norm: `0.008771423250436783`.
- Linear-DPO Frozen / EMA real-batch parity: `passed`.
- Linear loss: `-0.04078614339232445`.
- Linear grad norm: `0.5717411041259766`.
- EMA max absolute difference: `0.0`.

Outputs:

- `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260623/exp27_sdpo_real_batch_parity`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260623/exp27_linear_real_batch_parity`

Report:

`reports/exp27_gpu2_real_batch_parity.md`

No long training was launched.

## 2026-06-24 Nontrivial Parity and LocalDPO Smoke

The three-lane controller completed the requested nontrivial Exp27 gates:

- SDPO real-batch conflict case with `lambda_safe=0.314453125 < 1`.
- Linear-DPO Frozen / EMA multi-step parity with zero EMA discrepancy.
- LocalDPO six-video corruption pair and original loss 1/10-step smoke.

The LocalDPO official mask digest is still blocked by missing official code in
the local paper cache, so this remains a plumbing/original-loss smoke rather
than an official LocalDPO reproduction. RC-FPO was not started.

Report:

`reports/exp27_nontrivial_parity_and_localdpo_smoke_20260624.md`

## 2026-06-24 Distribution Scan and LocalDPO Official Path Fix

Status:

- `NONTRIVIAL_SDPO_PARITY_PASSED`
- `LINEAR_MULTISTEP_PARITY_PASSED`
- `LOCALDPO_SMOKE_PASSED`
- `SDPO_REAL_RESIDUAL_PROXY_SCAN_COMPLETE`
- `FAITHFUL_LOCALDPO_OFFICIAL_MASK_DIGEST_PASSED`
- `OBJECTIVE_STUDY_PENDING`
- `RCFPO_NOT_STARTED`

The SDPO scan used 32 real Gate32 preference rows and 4 sampled frames per row
for 128 real-video residual-proxy records. This is not a full DiffuEraser
policy-forward scan, but it avoids manually constructed gradient conflicts.

Results:

- `lambda_safe < 1` ratio: `0.4453125`
- lambda min: `0.2246925`
- lambda mean: `0.8942396`
- lambda max: `1.0`
- unsafe tiny-step winner-change rate: `0.0`

The LocalDPO official mask path was fixed to search commit-suffixed caches on
PAI. Official mask digest now passes from:

`/mnt/nas/hj/video_dpo_paper_code_cache/Local-DPO_7528e966b17283cfa638577827e456737335f030/innerT2V/utils/random_mask_gen.py`

RC-FPO, LocalDPO four-grid runs, and O0-O5 objective studies remain unstarted.

Reports:

- `reports/exp27_sdpo_real_distribution_scan.md`
- `reports/exp27_localdpo_official_path_fix.md`

## 2026-06-24 True-Model Forward Readback

Status:

- `TRUE_MODEL_FORWARD_READBACK_COMPLETE`
- `SDPO_REAL_RESIDUAL_PROXY_ONLY`
- `OBJECTIVE_STUDY_PENDING`
- `RCFPO_NOT_STARTED`

The latest readback re-confirmed that the current 128-record SDPO distribution
scan is a real-video residual proxy, not a full DiffuEraser policy/reference
forward. It does not load the policy/reference model, does not produce real
`model_pred`/`ref_pred`, and cannot promote RC-FPO.

Next allowed Exp27 milestone is the true DiffuEraser Stage1 policy/reference
forward scan.

Report:

`reports/exp27_true_model_forward_readback.md`
## 2026-06-25 PAI Pre-Maintenance Persistence

Status: `BLOCKED_NAS_PERMISSION`

Before launching true DiffuEraser policy/reference forward parity, the required
cross-track PAI `/home` artifact persistence gate was attempted. It is blocked
because SSH user `hj` cannot write to
`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch`; `sudo -n` and root
SSH are unavailable through the current session.

No new Exp27 GPU task was launched after this blocker. Exp27 remains at
`TRUE_MODEL_FORWARD_READBACK_COMPLETE`, with SDPO distribution results still
classified as `REAL_VIDEO_RESIDUAL_PROXY_ONLY`.

Report: `reports/pai_premaintenance_output_persistence.md`

## 2026-06-25 PAI Pre-Maintenance Persistence Resolved

Status:

- `PAI_PREMAINTENANCE_PERSISTENCE_PASSED`
- `TRUE_MODEL_FORWARD_READBACK_COMPLETE`
- `SDPO_REAL_RESIDUAL_PROXY_ONLY`
- `OBJECTIVE_STUDY_PENDING`
- `RCFPO_NOT_STARTED`

The NAS permission blocker was resolved from the PAI WebIDE root terminal by
granting `hj` write access to the required autoresearch/runtime directories.
HAL SSH as `hj` verified write access. The cross-track Exp25 Gate32 dense review
artifacts and Exp26 Gate64 official generation artifacts were copied to NAS
with matching file counts, byte totals, inventory diffs, and SHA256 diffs.

No Exp27 GPU task was started during persistence. The next allowed Exp27
milestone remains true DiffuEraser Stage1 policy/reference forward scan.

Report:

`reports/pai_premaintenance_output_persistence.md`

## 2026-06-25 SDPO Real Residual-Proxy Distribution Scan

Status:

- `SDPO_REAL_RESIDUAL_PROXY_SCAN_COMPLETE`
- `OBJECTIVE_STUDY_PENDING`
- `RCFPO_NOT_STARTED`

Exp27 scanned 32 real preference rows with four sampled timesteps/frames per
row, producing 128 residual-proxy records. This uses real VOR preference data
and real DiffuEraser Gate32 raw OR candidates, but it does not load
DiffuEraser policy/reference models and is not a full policy-forward gradient
distribution scan.

Results:

- `lambda_safe < 1` ratio: `0.4453125`
- lambda min/mean/max: `0.2246925 / 0.8942396 / 1.0`
- unsafe tiny-step winner-change rate: `0.0`

This supports continuing the true policy-forward scan, but it does not allow
RC-FPO, LocalDPO four-grid runs, or O0-O5 objective training to start.

Reports:

- `reports/exp27_sdpo_real_distribution_scan.md`
- `reports/exp27_sdpo_real_distribution_scan.csv`
- `reports/exp27_sdpo_real_distribution_scan_summary.json`

## 2026-06-25 PAI Post-Maintenance Permission Recovery

Status:

- `PAI_POSTMAINTENANCE_PERMISSIONS_RECOVERED`
- `BLOCKER_RESOLVED`
- `TRUE_MODEL_SDPO_READY_TO_RUN`
- `OBJECTIVE_STUDY_PENDING`
- `RCFPO_NOT_STARTED`

The user executed the minimal PAI root permission repair successfully. HAL SSH
as `hj` rechecked the actual PAI state and confirmed:

- DiffuEraser converted weights are readable/executable.
- Exp27 experiment output is writable.
- Exp27 autoresearch output is writable.

No ACL installation, apt operation, or broad NAS permission change was repeated
by Codex. The next allowed Exp27 milestone is true DiffuEraser policy/reference
SDPO forward parity.

Reports:

- `reports/exp27_permission_recovery_readback.md`
- `reports/pai_postmaintenance_permission_recovery_final.md`
- `reports/pai_postmaintenance_permission_recovery_final.csv`

## 2026-06-25 True DiffuEraser Policy/Reference SDPO Gate

Status:

- `TRUE_MODEL_PARITY`
- `SDPO_TRUE_MODEL_32X4_SCAN_COMPLETE`
- `SDPO_TINY_STEP_ACTUAL_CHECK_PASSED`
- `LINEAR_TRUE_MODEL_PROBE_PASS`
- `LINEAR_TRUE_MODEL_1_10_STEP_PENDING`
- `LOCALDPO_24F_PENDING`
- `OBJECTIVE_STUDY_PENDING`
- `RCFPO_NOT_STARTED`

Exp27 completed the real DiffuEraser Stage1 policy/reference forward gate using
SFT-48000 as the S0 policy/reference identity state and Exp11 outer b0.75
Stage1 as the S1 representative trained policy with frozen SFT-48000 reference.
The scan covered 32 real BR preference rows and four fixed timesteps per row.

Results:

- Total true-model records: `256`.
- S1 records: `128`.
- S1 `lambda_safe < 1`: `32 / 128 = 0.25`.
- SDPO lambda max abs diff vs extracted official helper: `0.0`.
- SDPO loss max abs diff: `0.0`.
- Output-gradient cosine min: `0.9999998807907104`.
- Actual tiny-step cases: `8` (`4` lambda<1 and `4` lambda=1).
- Tiny-step reference grad norm max: `0.0`.
- Tiny-step max policy parameter delta norm: `0.0009144449931267993`.
- Tiny-step mean margin change: `0.0003423636662773788`.

The Linear-DPO check in this milestone is a true-model-record probe over the
same real forward records, not the requested full Linear Frozen/EMA 1/10-step
training gate. That remains pending and must not be reported as training pass.

Reports:

- `reports/exp27_sdpo_true_model_forward_parity.md`
- `reports/exp27_sdpo_true_model_distribution_scan.csv`
- `reports/exp27_sdpo_true_model_summary.json`
- `reports/exp27_sdpo_true_model_tiny_step_cases.csv`
- `reports/exp27_linear_true_model_parity.md`
- `reports/exp27_linear_true_model_parity.csv`

## 2026-06-25 True Linear-DPO 1/10-Step Gate

Status:

- `LINEAR_TRUE_MODEL_1_10_STEP_PASSED`
- `LINEAR_FROZEN_TRUE_MODEL_10STEP_PASSED`
- `LINEAR_EMA_TRUE_MODEL_10STEP_PASSED`
- `TECHNICAL_PASS`
- `LOCALDPO_24F_PENDING`
- `OBJECTIVE_STUDY_PENDING`
- `RCFPO_NOT_STARTED`

Exp27 ran a true DiffuEraser Stage1 Linear-DPO micro gate on one fixed real BR
preference batch. This used the same real model-loading path as the SDPO true
model gate, SFT-48000 policy/reference initialization, shared real data,
shared timestep `500`, and no synthetic gradient conflict.

Results:

- variants: `linear_frozen`, `linear_ema`
- steps per variant: `10`
- Linear-Frozen max grad norm: `0.48048678696353975`
- Linear-Frozen step10 policy delta norm: `0.0012969709135074255`
- Linear-Frozen step10 reference delta norm: `0.0`
- Linear-EMA max grad norm: `0.49775458360972635`
- Linear-EMA step10 policy delta norm: `0.0013002275364513564`
- Linear-EMA step10 reference delta norm: `1.819002953296671e-08`
- NaN/Inf: `0`

This is a technical true-model micro-training pass. It does not start RC-FPO,
does not start 50-step/O0-O5 objective studies, and does not claim video
quality improvement.

Reports:

- `reports/exp27_linear_true_model_10step_readback.md`
- `reports/exp27_linear_true_model_10step.md`
- `reports/exp27_linear_frozen_10step.csv`
- `reports/exp27_linear_ema_10step.csv`

## 2026-06-25 CLI4 LocalDPO DiffuEraser 24F Adaptation Prep

Status:

- `LOCALDPO_24F_CLI4_READY_NOT_LAUNCHED`
- `P8_PENDING`
- `P32_PENDING`
- `LOCALDPO_ORIGINAL_OBJECTIVE_1_10_STEP_PENDING`
- `RCFPO_NOT_STARTED`

CLI4 added an isolated LocalDPO DiffuEraser 24F runner:

`exp27_paper_grounded_preference_study/scripts/run_exp27_localdpo_24f_adaptation.py`

The runner preserves the required sequence:

- official LocalDPO 3D moving corruption masks;
- real clean winner frames;
- SFT DiffuEraser self-model loser generation;
- outside reinjection/composite after loser generation;
- P8 gate before P32;
- P32 gate before objective;
- original LocalDPO-style `RA-DPO + global DPO + SFT` 1-step and 10-step
  micro objective only.

Controlled corruption previews may be used only for tests and are explicitly
non-gate-valid. This preparation does not start 50-step, four-grid 50-step,
O0-O5 objective study, or RC-FPO.

Reports:

- `reports/exp27_localdpo_24f_cli4_prelaunch.md`

## 2026-06-25 CLI4 LocalDPO 24F Safety-Checker Retry Fix

Status:

- `LOCALDPO_24F_CLI4_RETRY1_PREPARED`
- `P8_PENDING`
- `P32_PENDING`
- `LOCALDPO_ORIGINAL_OBJECTIVE_1_10_STEP_PENDING`
- `RCFPO_NOT_STARTED`

The first CLI4 LocalDPO 24F launch stopped during P8 DiffuEraser loser
generation because the OR path attempted to load
`stable-diffusion-v1-5/safety_checker/config.json`, which is absent from the
local PAI weight mirror. The non-OR DiffuEraser path already disables the
safety checker for local research inference; CLI4 applied the same isolated
arguments to `diffueraser/diffueraser_OR.py`.

This is a runtime compatibility fix only. It does not change the LocalDPO
objective, does not start O0-O5, does not start 50-step training, and does not
start RC-FPO.

Report:

- `reports/exp27_localdpo_24f_cli4_safety_checker_retry_fix.md`
