# Exp27 True Model Forward Readback

Date: 2026-06-24 UTC

## Git State

- Branch: `research/exp27-paper-grounded-preference-study`
- HEAD: `1a2afd626e815af427adf23eb1b446cd26fd03f4`
- Remote checked: `origin/research/exp27-paper-grounded-preference-study`
- Worktree status before this report: clean

## Files Read

PRD and registry:

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/49_exp27_paper_grounded_preference_study.md`
- `experiment_registry/exp27_paper_grounded_preference_study/status.md`

Reports:

- `reports/exp27_gpu2_real_batch_parity.md`
- `reports/exp27_nontrivial_parity_and_localdpo_smoke_20260624.md`
- `reports/exp27_sdpo_real_distribution_scan.md`
- `reports/exp27_localdpo_official_path_fix.md`
- `reports/exp27_localdpo_official_path_fixed_smoke.json`

Exp27 code:

- `exp27_paper_grounded_preference_study/code/official_parity.py`
- `exp27_paper_grounded_preference_study/scripts/run_exp27_real_batch_parity.py`
- `exp27_paper_grounded_preference_study/scripts/scan_sdpo_real_distribution.py`
- `exp27_paper_grounded_preference_study/code/localdpo_full_adapter.py`

Pinned official code:

- LocalDPO commit: `7528e966b17283cfa638577827e456737335f030`
- Diffusion-SDPO commit: `84fb241c1b89705a247da8b0d6047798ca49830d`
- Linear-DPO commit: `663179c7adbbbd2d77b97b5841534447eb291ebd`

## Completed Work Not Repeated

- SDPO scalar safe-lambda helper parity.
- SDPO tensor-shaped real-batch smoke.
- Constructed nontrivial SDPO conflict case with `lambda_safe=0.314453125`.
- Linear-DPO Frozen / EMA primitive and multi-step parity.
- LocalDPO official mask digest path fix.
- LocalDPO six-video original-loss smoke.

These remain valid technical gates, but they are not promoted to a full
DiffuEraser policy/reference forward distribution study.

## Current Limitation

The existing distribution scan is explicitly a real-video residual proxy:

- It uses real Gate32 rows and real video residuals.
- It does not load DiffuEraser policy/reference models.
- It does not compute real `model_pred` and `ref_pred`.
- It cannot be used to start RC-FPO.

Current residual-proxy result:

- Records: `128`
- `lambda_safe < 1` ratio: `0.4453125`
- lambda min / mean / max: `0.2246925 / 0.8942396 / 1.0`

## Pending True-Model Gate

Next required milestone is a true DiffuEraser Stage1 policy/reference forward
scan:

- Load real DiffuEraser Stage1 policy and frozen reference.
- Use real preference rows, masks, VAE latents, shared noise, and shared
  timestep.
- Record real `model_pred` and `ref_pred`.
- Compare Exp27 SDPO helper against the pinned official SDPO formula.
- Scan first `8 rows x 4 timesteps`, then `32 rows x 4 timesteps` only if the
  technical gate passes.

Required report outputs:

- `reports/exp27_sdpo_true_model_forward_parity.md`
- `reports/exp27_sdpo_true_model_distribution_scan.csv`
- `reports/exp27_sdpo_true_model_summary.json`

## Promotion Gate

Do not start RC-FPO or objective-study training until all direct baselines are
complete:

- true SDPO model parity;
- true Linear-DPO model parity;
- faithful LocalDPO 24-frame adaptation;
- LocalDPO four-cell study;
- O0-O5 objective study with search-dev video review.

Current status remains:

`OBJECTIVE_STUDY_PENDING`

`RCFPO_NOT_STARTED`
