# Exp26 VideoPainter Primary-32 L0/L1

Status: `VP_L0_L1_PASSED`

- PAI output: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp26_videopainter_dpo_v2/vp_l0_l1_primary32_retry3_20260625_125902`
- Log: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/vp_l0_l1_primary32_retry3_20260625_125902.log`
- Manifest: `/home/hj/runtime_code/H20_Video_inpainting_DPO_exp26_videopainter_run/exp26_videopainter_dpo_v2/manifests/vp2_gate64_primary32_final.jsonl`
- Sample: `vp2_gate64_012_BLENDER_BEACH028_00003`
- Frames: `49`
- CUDA_VISIBLE_DEVICES: `0`

## Official Optimizer/Scheduler

```json
{
  "adam_beta1": 0.9,
  "adam_beta2": 0.95,
  "adam_epsilon": 1e-08,
  "learning_rate": 0.0001,
  "lr_scheduler": "constant",
  "lr_warmup_steps": 500,
  "max_grad_norm": 1.0,
  "mixed_precision": null,
  "weight_decay": 0.0001
}
```

## L0

- status: `passed`
- loss: `0.695064902305603`
- DPO loss: `0.6931471824645996`
- policy grad norm: `14.379858399808493`
- reference has grad: `False`
- winner/ref gap: `0.0`
- loser/ref gap: `0.0`

## L1

- status: `passed`
- checkpoint: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp26_videopainter_dpo_v2/vp_l0_l1_primary32_retry3_20260625_125902/one_step_run/checkpoint-1`
- last weights: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp26_videopainter_dpo_v2/vp_l0_l1_primary32_retry3_20260625_125902/one_step_run/last_weights`
- policy delta norm: `1.6732795822542237`
- reference delta norm: `0.0`
- strict reload max abs diff: `0.0`
- reload loss: `2.0078787803649902`
- saved/reloaded digest match: `True`

Conclusion: the final primary-32 manifest can drive a real VideoPainter policy/reference DPO batch and a one-step optimizer update with strict checkpoint reload. This is a technical training gate only, not a quality-positive result.
