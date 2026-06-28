# Exp40 MiniMax PSNR-Safe Rescue

Status: `EXP40_READBACK_COMPLETED`

Branch: `research/exp40-minimax-psnr-safe-rescue-20260628`

Base: `origin/research/exp38-minimax-full-adapter-breakthrough-20260628`

## Scope

Exp40 continues MiniMax only. The goal is to rescue the small Exp38 R1 PSNR
signal while making boundary, outside, LPIPS, Ewarp, and visual quality safe.

Forbidden:

- use GPU2-GPU7;
- repeat Exp38 R1/R2/R3 exactly;
- use VOR-Eval for training, selection, or thresholding;
- use hard comp as primary evaluation;
- modify `inference/metrics.py`;
- modify shared trainer;
- rewrite Exp1-Exp38 history or overwrite Exp38 outputs;
- claim universal adapter, final SOTA, or top-conference novelty.

## 2026-06-28 Readback And R1 Positive-Signal Audit

Milestone A status: `EXP40_READBACK_COMPLETED`.

Git start:

- start HEAD: `06b17c0a4be2cb82d1ffbdf7b6c93406f37a3ff8`
- source commit: Exp38 `Run Exp38 MiniMax SFT-DPO rescue gate`

GPU0/GPU1 policy:

- PAI GPU0/GPU1 were audited.
- A stale Exp30 GPU0 heartbeat/bash process group was recorded and terminated:
  PGID `1715134`; command was the old Exp30 MiniMax gate64 launcher; no compute
  PID existed before cleanup.
- GPU0/GPU1 after cleanup: `0 MiB`, `0%`, no compute PID.
- GPU2-GPU7 were not used or signaled.
- audit path:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp40_minimax_psnr_safe_rescue/gpu0_1_preclean_exp30_stale_20260629_030722.txt`

Exp38 R1 signal:

- R1 full/mask/boundary/outside PSNR deltas:
  `+0.102167` / `+0.117230` / `-0.141510` / `-0.037262`.
- R1 positive full PSNR rows: `7/13`.
- R1 positive mask PSNR rows: `7/13`.
- R1 boundary-negative rows: `9/13`.
- R1 outside-negative rows: `8/13`.
- R1 outside-MAE-worse rows: `10/13`.
- R1 visual classification:
  `0/13` clear better, `9/13` worse/tradeoff, `4/13` tie/local numeric gain.

Answers:

1. R1 had `+0.102` full PSNR because several rows moved local mask content
   toward the winner. It failed visually because the movement was often
   over-erasure, fogging, or boundary/outside damage instead of clean object or
   effect removal.
2. Improved rows by numeric full/mask include `REAL_ENV103_00001_001_01`,
   `REAL_ENV089_00001_001_01`, `REAL_ENV093_00001_001_01`,
   `REAL_ENV103_00004_001_01`, `REAL_ENV095_00002_001_01`,
   `REAL_ENV104_00001_001_01`, and `REAL_ENV105_00004_001_01`.
3. R1 train-overfit did not establish clean train improvement either: earlier
   Exp38 train32 had full/outside regression with local movement.
4. Boundary loss is a primary blocker: R1 boundary PSNR is negative in `9/13`
   rows and mean boundary delta is `-0.141510`.
5. Outside damage is also a blocker: outside PSNR is negative in `8/13` rows
   and outside MAE worsens in `10/13`.
6. R1 likely used useful local Linear-DPO/hard-noise pressure but insufficient
   boundary/outside preservation; the issue is not simply "more DPO".
7. Keep from R1: frozen-reference local Linear-DPO, hard-state evaluation,
   and the observation that MiniMax output can move in the right raw-PSNR
   direction.
8. Do not repeat R2/R3: SDPO-safe and SFT-warmup variants from Exp38 produced
   larger harmful drift and negative aggregate boundary/outside metrics.
9. Exp40 success target: shadow raw full PSNR `> +0.2 dB`, mask PSNR
   improvement, boundary/outside nonnegative or safe, LPIPS/Ewarp safe, and
   visual review without fogging, over-erasure, boundary damage, or outside
   damage.
10. Readback files are listed in
   `reports/exp40_minimax_psnr_safe_readback.md`.

Next milestone: sample-level R1 diagnosis and recipe narrowing. No GPU training
is allowed until the readback commit is pushed.

## 2026-06-28 R1 Sample-Level Diagnosis

Milestone B status: `MINIMAX_R1_SIGNAL_AUDITED`.

No new training, inference, or GPU task was launched. This milestone used
existing Exp38/Exp37 R1 evidence only.

Availability caveat:

- Exp38 SFT/DPO rescue R1 wrote heldout13 outputs only.
- It did not write per-train R1 outputs under the R1 rescue output root.
- Train-side diagnosis therefore uses the existing Exp38 train-overfit audit of
  the Exp37 R1 checkpoint on train32/heldout16.
- Metrics not present in those prior reports, including LPIPS, Ewarp, affected
  PSNR, object residual, and effect residual, are explicitly marked
  `NOT_AVAILABLE` in the CSV rather than inferred.

Aggregate diagnosis:

- Exp38 SFT/DPO R1 heldout13: full/mask/boundary/outside means
  `+0.102167` / `+0.117230` / `-0.141510` / `-0.037262`.
- Existing train-overfit Exp37 R1 train32: full/mask/boundary/outside means
  `-0.586255` / `+0.152062` / `+0.069123` / `-0.895018`.
- Existing train-overfit Exp37 R1 heldout16: full/mask/boundary/outside means
  `+0.200826` / `+0.161946` / `-0.049755` / `+0.028198`.

Diagnosis counts across available rows:

- `R1_GOOD_BUT_OUTSIDE_BAD`: `28`
- `R1_GOOD_BUT_BOUNDARY_BAD`: `9`
- `R1_GOOD_LOCAL_IMPROVEMENT_NOT_DECISIVE`: `10`
- `R1_WORSE`: `9`
- `R1_TIE`: `4`
- `R1_FOGGING_OVER_ERASURE`: `1`

Decision:

R1 failures are dominated by outside/background drift, boundary cost, and local
over-erasure/fogging risk. Exp40 must not run a broader DPO search first. The
next recipe family must start with a larger, cleaner LocalDPO v3 pool and
PSNR-safe SFT with stronger boundary/outside preservation; DPO is allowed only
after SFT produces search/shadow-safe improvement.

Reports:

- `reports/exp40_r1_sample_level_diagnosis.md`
- `reports/exp40_r1_sample_level_diagnosis.csv`
- `reports/exp40_r1_visual_review.csv`
- `reports/exp40_r1_diagnosis_summary.json`

## 2026-06-29 LocalDPO v3 PSNR-Safe Pool

Milestone C status: `MINIMAX_LOCALDPO_V3_POOL_READY_MINIMUM`.

No MiniMax training was launched. GPU0/GPU1 remained unused by this milestone
after the earlier stale Exp30 process-group cleanup; GPU2-GPU7 were untouched.

Pool construction:

- Source: VOR-Train only.
- VOR-Eval used: `false`.
- Hard comp used: `false`.
- Condition: `V_obj`.
- Winner: `V_bg`.
- Loser: locally corrupted `V_bg`.
- Candidate rows: `336`.
- Candidate rows per source: `<= 3`.
- Selected split counts: `train=64`, `search=24`, `shadow=24`.
- Selected source balance:
  - train: `BLENDER=32`, `REAL=32`
  - search: `BLENDER=12`, `REAL=12`
  - shadow: `BLENDER=12`, `REAL=12`
- Selected classification: all `112/112` rows are
  `MEDIUM_HARD_ELIGIBLE`.
- Scene overlap:
  - train/search: `0`
  - train/shadow: `0`
  - search/shadow: `0`

Target caveat:

- The requested target was `train96/search32/shadow32`.
- The run reached the pre-registered minimum `train64/search24/shadow24`, not
  the full target.
- This is enough to continue with small Step0/SFT diagnostics, but later paper
  claims must keep the minimum-pool caveat.

Extraction / materialization:

- A full tar extraction attempt was stopped before completion because the
  gzip-tar member scan was slow and an existing exact materialized cache was
  available.
- The selected pool uses existing materialized refs plus the exact cache:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/extracted/vor_train_audit64_exact_20260623`.
- Metadata index SHA256:
  `33d57a3ea23c5799b583d476a311089f95cbce1b0d11280822a63b8c9edcddc4`.

Visual review:

- Codex opened all 19 temporal-strip review pages covering all 112 selected
  rows.
- Review status: `REVIEWED_PASS_TEMPORAL_STRIP_POOL_AUDIT`.
- Observed pattern: local object/affected-region perturbations, no global
  collapse, no black/purple failure, and no systematic far-outside damage in
  the review sheets.
- This is a data-pool construction pass only. It is not a model quality pass and
  does not imply MiniMax adapter success.

Output caveat:

- PAI `hj` could not write to the requested Exp40 experiments output root, so
  this milestone stores selected-pool outputs under the Exp40 log root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp40_minimax_psnr_safe_rescue/localdpo_v3_refs_exact_v3_20260629_042425`.

Reports and manifests:

- `reports/exp40_localdpo_v3_pool.md`
- `reports/exp40_localdpo_v3_pool.csv`
- `reports/exp40_localdpo_v3_visual_review.csv`
- `reports/exp40_localdpo_v3_summary.json`
- `reports/exp40_localdpo_v3_review_pages/`
- `exp40_minimax_psnr_safe_rescue/manifests/exp40_localdpo_v3_train96.jsonl`
- `exp40_minimax_psnr_safe_rescue/manifests/exp40_localdpo_v3_search32.jsonl`
- `exp40_minimax_psnr_safe_rescue/manifests/exp40_localdpo_v3_shadow32.jsonl`
- `exp40_minimax_psnr_safe_rescue/manifests/exp40_localdpo_v3_rejected.jsonl`
