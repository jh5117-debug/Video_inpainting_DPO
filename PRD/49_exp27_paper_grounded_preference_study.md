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
