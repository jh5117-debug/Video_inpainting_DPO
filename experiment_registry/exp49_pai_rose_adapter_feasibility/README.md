# Exp49 PAI ROSE Adapter Feasibility

This registry tracks ROSE as a possible third adapter candidate.

Current role: `ROSE_BASELINE_READY__ROSE_LOSER_GENERATOR_USEFUL__ROSE_TRAINING_FORWARD_BLOCKED`.

ROSE official inference is runnable on PAI and Gate16 passed as a VOR-Train baseline/loser-generator screen. Gate16 produced `16/16` decodable rows, with visual labels `ROSE_OUTPUT_USABLE=9`, `MEDIUM_HARD_ELIGIBLE=5`, `SIDE_EFFECT_LEFT=2`, and `TRIVIAL_BAD=0`.

Do not interpret this registry as ROSE adapter evidence. Milestone D found no released official train/finetune optimizer/backward/loss/target script. One-step and 10-step adapter gates remain locked.

## Key Reports

- `reports/exp49_rose_asset_download.md`
- `reports/exp49_rose_env_smoke.md`
- `reports/exp49_rose_code_adapter_feasibility_audit.md`
- `reports/exp49_rose_official_inference_smoke.md`
- `reports/exp49_rose_vor_or_gate16.md`
- `reports/exp49_rose_feasibility_outcome.md`
- `reports/exp49_rose_paper_positioning.md`
