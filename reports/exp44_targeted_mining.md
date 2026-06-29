# Exp44 Targeted MiniMax Same-Source Mining

Status: `MINIMAX_TARGETED_MINING_COMPLETED`

This milestone aggregates official MiniMax raw inference candidates from
the targeted second-pass workers. Automatic labels are provisional; the
next milestone must perform strict visual relabeling before any same-source
pair or Stage2 handoff manifest is trusted.

## Counts

- Candidates: `452`
- Auto successful-removal candidates: `138`
- Auto medium-hard failure candidates: `231`
- Auto same-source pair capacity: `26`
- Auto overlap groups: `13`

## Guardrails

- Training run: `false`
- VOR-Eval used: `false`
- Hard comp used: `false`
- Raw output primary: `true`

## Artifacts

- All candidates: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp44_pai_minimax_targeted_same_source_mining/targeted_mining_20260629_161742/manifests/exp44_targeted_candidates_all.jsonl`
- Auto success: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp44_pai_minimax_targeted_same_source_mining/targeted_mining_20260629_161742/manifests/exp44_targeted_success_auto.jsonl`
- Auto failure: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp44_pai_minimax_targeted_same_source_mining/targeted_mining_20260629_161742/manifests/exp44_targeted_failure_auto.jsonl`
- Metrics CSV: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp44_pai_minimax_targeted_same_source_mining/targeted_mining_20260629_161742/reports/exp44_targeted_mining_metrics.csv`
- Group yield CSV: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp44_pai_minimax_targeted_same_source_mining/targeted_mining_20260629_161742/reports/exp44_targeted_mining_group_yield.csv`
