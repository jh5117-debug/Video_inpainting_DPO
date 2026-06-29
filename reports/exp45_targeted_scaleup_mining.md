# Exp44 Targeted MiniMax Same-Source Mining

Status: `MINIMAX_TARGETED_MINING_COMPLETED`

This milestone aggregates official MiniMax raw inference candidates from
the targeted second-pass workers. Automatic labels are provisional; the
next milestone must perform strict visual relabeling before any same-source
pair or Stage2 handoff manifest is trusted.

## Counts

- Candidates: `72`
- Auto successful-removal candidates: `38`
- Auto medium-hard failure candidates: `26`
- Auto same-source pair capacity: `16`
- Auto overlap groups: `5`

## Guardrails

- Training run: `false`
- VOR-Eval used: `false`
- Hard comp used: `false`
- Raw output primary: `true`

## Artifacts

- All candidates: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp45_pai_minimax_pair_scaleup/scaleup_mining_20260629_190229/manifests/exp44_targeted_candidates_all.jsonl`
- Auto success: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp45_pai_minimax_pair_scaleup/scaleup_mining_20260629_190229/manifests/exp44_targeted_success_auto.jsonl`
- Auto failure: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp45_pai_minimax_pair_scaleup/scaleup_mining_20260629_190229/manifests/exp44_targeted_failure_auto.jsonl`
- Metrics CSV: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp45_pai_minimax_pair_scaleup/scaleup_mining_20260629_190229/reports/exp44_targeted_mining_metrics.csv`
- Group yield CSV: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp45_pai_minimax_pair_scaleup/scaleup_mining_20260629_190229/reports/exp44_targeted_mining_group_yield.csv`
