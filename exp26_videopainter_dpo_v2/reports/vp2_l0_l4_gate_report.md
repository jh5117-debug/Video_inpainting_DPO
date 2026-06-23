# Exp26 VideoPainter v2 L0-L4 Gate Report

status: passed
cuda_visible_devices: 2
formal_manifest: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp26_videopainter_dpo_v2/l0_l4_gates_20260623_110548/gate_data/formal_49f/bear/formal_49f_manifest.jsonl`
plumbing_manifest: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4/manifests/selected_primary_comp.repaired.pai_paths.jsonl`

## Official Optimizer/Scheduler
- adam_beta1: 0.9
- adam_beta2: 0.95
- adam_epsilon: 1e-08
- learning_rate: 0.0001
- lr_scheduler: constant
- lr_warmup_steps: 500
- max_grad_norm: 1.0
- mixed_precision: None
- weight_decay: 0.0001

## Gates
- L0: passed
  - native_shape: [1, 13, 16, 60, 90], checksum16: 04d5ffbe7d8de8ef
- L1: passed
  - max_abs_diff: 3.039836883544922e-06, dpo_loss: 0.6931471824645996
- L2: passed
  - dpo_loss: 0.6931471824645996, abs_gap_to_log2: 1.904654323148236e-09
- L3: passed
  - checkpoint: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp26_videopainter_dpo_v2/l0_l4_gates_20260623_110548/formal_run/checkpoint-1`; state_max_abs_diff_after_reload: 0.0; output_delta_after_update: 25.649038111791015; reload_output_delta: 0.0
- L4: passed
  - label: PLUMBING_ONLY_13F; steps: 10; checkpoint: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp26_videopainter_dpo_v2/l0_l4_gates_20260623_110548/plumbing_13f_run/checkpoint-10`; last_loss: 0.0039273826405406; last_dpo_loss: 0.0015321369282901287

L4 is explicitly `PLUMBING_ONLY_13F`; formal 49-frame preference data construction is now unblocked, but no long training was started.
