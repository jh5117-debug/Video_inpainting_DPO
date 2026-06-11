# Corrections To User Experiment Summary

| item | judgment | corrected_statement | evidence_path |
| --- | --- | --- | --- |
| Exp4 | partially correct | win=VideoDPO win; lose=DiffuEraser full-mask generated video; task 仍是 full-mask bridge；plain DPO 是计划/记录方向，但该数据 gate 失败且后续数据删除。 | experiment_registry/exp04_fullmask_loser_failed_gate/config.yaml |
| Exp5 | partially correct | win=D2/VideoDPO manifest winner；lose=D2 selected-primary comp loser；registry 明确是 d2_comp_k4，mask 来自 D2 partial mask，但训练用 full-mask bridge。它不是单纯 full-mask generated loser；comp 对 loser asset 有意义。 | experiment_registry/exp05_old_plain/config.yaml |
| NewExp5 | correct | 同 D2 comp 数据，但 loss 改为 beta=10、lose_gap_weight=0.25、winner_abs=0.05、winner_gap=1.0。 | experiment_registry/exp05_new_wingap_comp/config.yaml |
| NewExp6 | correct | 同 NewExp5 loss，数据改为 D2 no-comp/raw loser。不要写成 plain Exp6。 | experiment_registry/exp06_new_wingap_nocomp/config.yaml |
| Exp7a | correct with caveat | D2 partial-mask task；win=D2 manifest winner；lose=D2 comp partial-mask loser；loss 与 winner-anchor DPO 一致。caveat: 早期 val 协议有 DAVIS/small-D2 混用。 | experiment_registry/exp07a_1_stage1dpo_sftstage2/config.yaml |
| Exp8a | correct | win=generated high-score rollout winner，非 GT；lose=D3 generated loser；YouTube-VOS/D3 target-domain 数据；winner-anchor DPO。 | experiment_registry/exp08a_1_stage1dpo_sftstage2_davis/config.yaml |
| Exp8c | correct | win=GT/clean YouTube-VOS clip；lose=D3 generated loser；winner-anchor DPO。 | experiment_registry/exp08c_1_gtwin_stage1dpo_sftstage2_davis/config.yaml |
| Exp9 | correct | GT-win + D3 generated loser；log-ratio normalized gap + clipped loser gap + winner anchor。 | experiment_registry/exp09_1/config.yaml |
| Exp10 | correct | Exp9 基础上加入 region-local weighted MSE。 | experiment_registry/exp10_1/config.yaml |
| Exp11 | correct | 当前只能叫 Exp11-proxy：frozen-ref prior + boundary + temporal residual proxy；不是 real flow-prior / RAFT / ProPainter image-space prior。 | experiment_registry/exp11_flow_prior_consistency_dpo/status.md |
