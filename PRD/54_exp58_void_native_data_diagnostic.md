# Exp58 VOID Native Data Diagnostic

Date: 2026-07-02

Branch: `research/exp58-void-native-data-diagnostic-20260702`

Base: `origin/research/exp57-void-adaptive-transition-core-20260701`

## Goal

Exp58 tests whether VOID's remaining adapter failure is primarily a data-distribution mismatch rather than another loss tweak problem. Exp50-Exp57 established that VOID official inference, preference forward, zero-gap, one-step plumbing, and same-model loser generation work, but every VOR-derived one-step rescue remains mixed or negative because overlap, affected, and boundary regions regress.

## Hypothesis

VOID's official training data is generated paired counterfactual data, not VOR-derived object-removal data. The official release includes HUMOTO + Kubric generation code and `datasets/void_train_data.json` metadata, but not the rendered training videos in this repo checkout. VOR-derived quadmasks may not match VOID's native interaction-region semantics.

## Storage Policy

Large data should prefer PAI/NAS over H20 local storage. H20 `/home/nvme01` is allowed only for git worktree, isolated env, small staging, and short-lived render scratch.

Current audit:

- PAI `/home`: about 5T free.
- PAI `/mnt/nas`: very large and mounted.
- PAI requested experiment output parent `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo` is root-owned and not writable by `hj` for new Exp58 output root creation.
- PAI logs/runtime roots are writable.
- H20 `/home/nvme01`: about 1.2T free, 66% used; safe for tiny Gate8 smoke, not large accumulation.

## Allowed Work

- Add isolated code under `exp58_void_native_data_diagnostic/`.
- Set up an isolated Kubric env.
- Generate tiny Kubric Gate8 only if official dependencies and assets are available.
- Run official VOID inference and one-step diagnostics only if generated data is technically valid.

## Forbidden Work

- No 10-step or long training.
- No VOR-Eval.
- No hard comp.
- No VOID official source edits.
- No shared trainer or `inference/metrics.py` edits.
- No universal adapter, final SOTA, or third-backbone claim.

## Milestones

| Milestone | Status | Notes |
| --- | --- | --- |
| A | `EXP58_READBACK_DONE` | Data mismatch and storage readback completed. |
| Storage | `EXP58_STORAGE_PAI_NAS_PREFERRED` | PAI/NAS preferred; requested PAI experiment output root is not writable by `hj`. |
| B | pending | Isolated Kubric env smoke. |
| C | pending | Generate VOID-native Kubric Gate8 if env/assets allow. |
| D | pending | Official VOID inference on Kubric Gate8. |
| E | pending | Kubric preference forward / zero-gap / one-step. |
| F | pending | VOR-vs-Kubric comparison. |
| G | pending | Final decision. |

## Scientific Position

VOID remains a VOR-OR inference baseline, same-model loser-generator candidate, and adapter-engineering candidate. Exp58 may support `VOID_DATA_MISMATCH_SUSPECTED` or confirm a native-data path, but it cannot claim third-backbone evidence without a later one-step PASS and aggregator-approved 10-step positive result.
