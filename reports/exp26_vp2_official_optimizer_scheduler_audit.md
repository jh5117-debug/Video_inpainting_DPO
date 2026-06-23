# Exp26 VideoPainter Official Optimizer/Scheduler Audit

- official_train_file: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter/train/train_cogvideox_inpainting_i2v_video.py`
- locked_json: `reports/exp26_vp2_locked_official_optimizer_scheduler.json`

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
