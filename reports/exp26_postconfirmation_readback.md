# Exp26 Post-Confirmation Readback

- created_utc: `2026-06-26T01:10:21Z`
- branch: `research/exp26-videopainter-dpo-v2`
- HEAD: `dde67b6cad69a525e378e99ed37337a932f869b1`
- worktree: `/home/hj/H20_Video_inpainting_DPO_exp26_videopainter`
- current conclusion read from registry/reports: `VIDEOPAINTER_SHADOWDEV_CONFIRMED`
- cross-backbone claim read from registry/reports: `CROSS_BACKBONE_ADAPTER_EVIDENCE_CONFIRMED`

## Files Read

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/48_exp26_videopainter_dpo_v2.md`
- `experiment_registry/exp26_videopainter_dpo_v2/status.md`
- `experiment_registry/exp26_videopainter_dpo_v2/paths.yaml`
- `experiment_registry/exp26_videopainter_dpo_v2/config.yaml`
- `experiment_registry/exp26_videopainter_dpo_v2/metric_summary.md`
- `experiment_registry/exp26_videopainter_dpo_v2/qualitative_summary.md`
- `experiment_registry/exp26_videopainter_dpo_v2/results.tsv`
- `reports/exp26_vp_50step_final.md`
- `reports/exp26_vp_shadowdev_final_decision.md`
- `reports/exp26_vp_shadowdev_metrics_and_statistics.md`
- `reports/exp26_vp_shadowdev_statistics.json`
- `reports/exp26_vp_shadowdev_tc_vfid.md`
- `reports/exp26_vp_shadowdev_visual_review.md`
- `reports/exp26_vp_shadowdev_seed_robustness.md`
- `reports/exp26_vp_50step_dynamics_audit.md`
- `reports/exp26_gate64_manifest_identity.json`
- `reports/exp26_gate64_primary32_final.md`

## Code Read

- `exp26_videopainter_dpo_v2/code/run_vp2_gate64_official_generation.py`
- `exp26_videopainter_dpo_v2/code/train_videopainter_dpo_adapter.py`
- `exp26_videopainter_dpo_v2/code/materialize_vp2_49f_sources.py`
- `exp26_videopainter_dpo_v2/code/generate_vp2_moving_br_masks.py`
- `exp26_videopainter_dpo_v2/code/evaluate_vp2_step0_searchdev.py`
- `exp26_videopainter_dpo_v2/code/shadowdev_confirmatory_analysis.py`

## Locked Identities

- primary32 SHA256: `82f661f2f30a581a213972533817624217eabb97eba7aaeedc00ee2109e4e716`
- search-dev SHA256: `41c6571d26e4a5130818dd50fbbe1314c9d953284561a3cd20f630572f7c2a71`
- shadow-dev SHA256: `0338dba1513cfe0e5dd85cbf793b3782902b981ac9610b0e472c6a048f738c02`
- fixed trajectory: `vp_primary32_50step_20260625_171032`
- fixed primary checkpoint: `checkpoint-50`
- official Step0 checkpoint: VideoPainter branch checkpoint used by the same generation runner

## What Is Already Done

- Gate64 formal-valid and final temporal review.
- Final primary32 comp-loser manifest.
- Step0/1/10/50 micro-training and search-dev evaluation.
- Independent shadow-dev Step0/10/30/50 generation.
- Shadow-dev metric/statistical gate.
- Shadow-dev TC/VFID diagnostics.
- Shadow-dev 32/32 video review.
- Shadow-dev seed robustness.
- 50-step dynamics audit.

## Must Not Be Repeated

- No new VideoPainter training.
- No 100-step or longer run.
- No primary32/search-dev/shadow-dev reselection.
- No Step30/Step50 checkpoint reselection from shadow-dev or external validation.
- No RC-FPO.
- No inference/metrics.py or shared-trainer edits.
- No access or mutation of left CLI worktrees, runtime, locks, outputs, or processes.

## Left CLI Protection State

Read-only PAI audit found left CLI Exp28 processes and locks on GPU1-4:

- GPU1/2: `exp28_pairB_inner4`, wrapper PGID `302909`, worker PIDs `396007`, `396008`
- GPU3/4: `exp28_pairA_inner2`, wrapper PGID `260093`, worker PIDs `394884`, `394885`
- monitor PID: `258013`

Right-side Exp26 did not send signals and did not modify left-side files.

## This Round Milestones

1. Post-confirmation sanity audit.
2. External 49-frame clean-source inventory.
3. If enough external sources exist, preregister and run fixed Step0 vs Step50 external validation.
4. Build VideoPainter evidence pack.
5. Audit third-model adapter compatibility.

