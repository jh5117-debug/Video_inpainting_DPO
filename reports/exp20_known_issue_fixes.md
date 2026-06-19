# Exp20 Known Issue Fixes

## Scope

This report covers the first correctness gate before the real DiffuEraser Stage1 trainer is wired. Changes are isolated to `exp20_autoresearch_scale_adaptive_region_dpo/` code and tests plus Exp20 reports.

## 3.1 Config Hash Dedup

Fixed `TrialConfig.config_hash()` so non-result fields do not affect deduplication:

- excluded from hash: `trial_id`, `parent_id`, `description`, `checkpoint_path`, `log_path`, `gpu_ids`
- included in hash: result-affecting radius/boundary/aggregation/seed/stage/train budget, checkpoint/data/code/evaluator identities, and fixed DPO hyperparameters

Test:

- `tests/test_search_config_hash.py`

## 3.2 Adaptive Radius Per Clip

Added per-clip adaptive radius computation for `[B,T,H,W]` batches:

- returns `radius_px` with shape `[B]`
- ignores empty-mask frames during the median statistic
- uses explicit fallback for fully empty clips
- supports batch size 1 and batch size > 1 without global flattening
- records empty/clamp metadata

Test:

- `tests/test_per_clip_adaptive_radius.py`

## 3.3 Image-Space Stable Region Partition

For `fixed_image_px` and adaptive image-space modes, region maps now form a stable partition:

- construct image-space mutually exclusive `M_img`, `B_img`, `O_img`
- area-pool each region to loss resolution
- clamp and normalize so each loss cell satisfies `M_loss + B_loss + O_loss ~= 1`

The `legacy_latent_exact` path remains hard/nearest/3x3-max-pool and preserves the Exp11 latent semantics.

Tests:

- `tests/test_region_partition_of_unity.py`
- `tests/test_image_space_boundary_no_overlap.py`

## 3.4 Exp11 Loser-Dominant Diagnostics

Changed loser-dominant diagnostics from the incorrect `m_l > m_w` style to the Exp11 pair-level definition:

- `inside_term`
- `winner_improvement = max(0, -pair_win_gap)`
- `loser_degradation = max(0, pair_lose_gap_clipped)`
- `correct_mask = inside_term > 0`
- `loser_dominant = correct_mask AND loser_degradation > winner_improvement`

Added/kept fields required for cross-device gather:

- `loser_degrade_ratio`
- `loser_degrade_count`
- `n_correct`
- `n_total`
- `winner_improvement_mean`
- `loser_degradation_mean`
- `_inside_term`
- `_winner_improvement`
- `_loser_degradation`
- `_pair_raw_win_gap`
- `_pair_raw_lose_gap`
- `_pair_norm_win_gap`
- `_pair_norm_lose_gap`
- `_pair_norm_lose_gap_clipped`
- `_pair_m_w`
- `_pair_m_l`
- `_pair_m_w_ref`
- `_pair_m_l_ref`

Test:

- `tests/test_exp11_diagnostic_parity.py`

## 3.5 Exp11 Exact Config Audit

Not complete yet. This requires reading the final Exp11 launcher/registry/checkpoint identity and is scheduled before trainer parity.

Planned output:

- `reports/exp20_exp11_exact_config_audit.md`

## 3.6 Image-Space Distance Cache

Added a persistent distance-transform cache:

- cache key includes implementation version, caller identity, mask hash, and image shape
- cached value is reusable for fixed/adaptive radii
- reports hit/miss/build-time stats
- avoids CPU distance-transform work every training step once cached

Tests:

- `tests/test_distance_cache_identity.py`
- `tests/test_distance_cache_reload.py`

## Validation

Current validation on HAL:

- `python -m py_compile exp20_autoresearch_scale_adaptive_region_dpo/code/*.py`: passed
- `python -m unittest discover -s exp20_autoresearch_scale_adaptive_region_dpo/tests -p 'test_*.py'`: 16 tests passed
- `bash -n exp20_autoresearch_scale_adaptive_region_dpo/scripts/*.sh`: passed
- `git diff --check`: passed before this report was added

## Current State

`PRECHECK_IMPLEMENTED`

The real Stage1 trainer, legacy full parity, 10-step smoke, dev split, dev baselines, and first fixed-boundary PAI search have not been run yet.
