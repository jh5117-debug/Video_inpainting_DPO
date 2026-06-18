# Exp19 ProPainter Completed-Flow Cache Quality

- input_manifest: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4/manifests/selected_primary_comp.repaired.pai_paths.jsonl`
- output_manifest: `/mnt/nas/hj/H20_Video_inpainting_DPO/data/cache/exp19_propainter_completed_flow_limit100/manifests/exp19_train_with_completed_flow_limit100.jsonl`
- output_root: `/mnt/nas/hj/H20_Video_inpainting_DPO/data/cache/exp19_propainter_completed_flow_limit100`
- limit: `100`
- flow_input: `masked winner frames; mask interior zeroed before RAFT`
- confidence: `forward-backward consistency, no GT-error`

## Summary

- ok/reused: `100` / `100`
- mean flow_conf_mean: `0.478330`
- mean valid_flow_ratio: `0.633401`
- mean flow magnitude: `9.528358`
- mean forward_backward_error: `3.911941`

Training must not start if confidence collapses to zero, valid_flow_ratio is below 0.2,
or visualized flow direction/scale is obviously wrong.
