# VideoPainter Training Entry Audit

Status: training entry exists.

## Training Scripts

Local scripts:

```text
train/VideoPainter.sh
train/VideoPainterID.sh
train/train_cogvideox_inpainting_i2v_video.py
train/train_cogvideox_inpainting_i2v_video_resample.py
```

Official command style:

```text
accelerate launch --config_file accelerate_config_machine_single_ds.yaml \
  train_cogvideox_inpainting_i2v_video.py \
  --pretrained_model_name_or_path ../ckpt/CogVideoX-5b-I2V \
  --meta_file_path ../data/pexels_videovo_train_dataset.csv \
  --val_meta_file_path ../data/pexels_videovo_val_dataset.csv \
  --instance_data_root ../data/videovo/raw_video \
  --height 480 \
  --width 720 \
  --max_num_frames 49 \
  --train_batch_size 1 \
  --inpainting_loss_weight 1.0 \
  --mask_add
```

## Answers

1. Training script exists: yes.
2. Official command: `train/VideoPainter.sh` / `train/VideoPainterID.sh`.
3. Training data format: CSV metadata plus raw video folder and `all_masks.npz`
   under VideoPainter's `data/video_inpainting/...` layout.
4. Input includes video + mask: yes. The collator returns `pixel_values`,
   `conditioning_pixel_values`, and `masks`.
5. Output: branch / LoRA style checkpoints saved by Accelerate hooks.
6. Loss: velocity/latent MSE plus `inpainting_loss_weight * mask MSE`.
7. Diffusion timestep / noise prediction: yes. The loop samples timestep/noise,
   adds noise to latents, predicts velocity, and computes denoising loss.
8. Frozen reference model: not implemented in upstream, but possible by loading
   a second branch/transformer copy under `torch.no_grad()`.
9. Checkpoint save: yes, through `accelerator.save_state`.
10. DAVIS validation: upstream evaluation supports DAVIS layout.
11. YouTube-VOS training: possible only after converting our YouTube-VOS/D3
    manifest into VideoPainter CSV/raw-video/mask format.

## Key Code Locations

```text
train/train_cogvideox_inpainting_i2v_video.py:1537-1605  dataset / dataloader
train/train_cogvideox_inpainting_i2v_video.py:1774-1811  video/mask latents
train/train_cogvideox_inpainting_i2v_video.py:1826-1853  noise/timestep/noisy latents
train/train_cogvideox_inpainting_i2v_video.py:1857-1879  branch + transformer forward
train/train_cogvideox_inpainting_i2v_video.py:1885-1891  denoising + inpainting loss
train/train_cogvideox_inpainting_i2v_video.py:1912-1933  checkpoint save
```

## Decision

Training entry passes feasibility, but not adapter readiness. DPO adapter
requires copied/adapted training code under `exp14_adapter_videopainter/code/`.

