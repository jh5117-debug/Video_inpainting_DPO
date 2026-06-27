# Exp33 EffectErase VOR-Eval Baseline Status

Current status: `EXP33_VOREVAL_OFFICIAL81_RUNNER_READY`

- branch: `research/exp33-effecterase-vor-eval-baseline-20260627`
- base: `origin/research/exp29-minimax-effecterase-adapter-feasibility-20260626`
- base HEAD: `6bc6c67c60b5cf2fe8d937ffd1e1d88a4684991c`
- worktree: `/home/hj/H20_Video_inpainting_DPO_exp33_effecterase_eval`
- planned GPU: GPU3 for official EffectErase baseline inference only.
- training: forbidden.
- VOR-Eval official81 compatibility: 43/43 ready, 0 rejected.
- ready manifest SHA256:
  `d5dc6052aae897ff01dcc2af8209de51dfbd04caf3f37534f0940c1f11a94811`.
- preview sanity review: 3 stratified rows passed.
- materialization: 43/43 rows ready, 129 condition/winner/mask MP4 files
  written in the Exp33 NAS run root.
- materialized preview sanity review: 3 stratified rows passed.
- runner: `exp33_effecterase_vor_eval_baseline/scripts/run_effecterase_vor_eval_official81.py`
- runner policy: held-out VOR-Eval baseline only, no adapter/training.
- command validation: pending.
- inference: not started.

Current final-status family: `EFFECTERASE_BASELINE_ONLY_FOR_NOW`
