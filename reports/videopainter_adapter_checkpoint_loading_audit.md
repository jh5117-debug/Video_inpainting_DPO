# VideoPainter Adapter Checkpoint Loading Audit

- baseline_checkpoint: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter/ckpt/VideoPainter/checkpoints/branch`
- adapter_checkpoint: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/exp14_adapter_videopainter/runs/gate2000/last_weights`
- baseline_weight: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter/ckpt/VideoPainter/checkpoints/branch/diffusion_pytorch_model.safetensors`
- adapter_weight: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/exp14_adapter_videopainter/runs/gate2000/last_weights/branch/diffusion_pytorch_model.safetensors`
- baseline_weight_exists: `True`
- adapter_weight_exists: `True`
- adapter_load_path: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/exp14_adapter_videopainter/runs/gate2000/last_weights`
- fallback_used: `False`
- baseline_sha256_head: `a8e5e02dae3c8f2d46df5b5cead0947fa84092c6958dc83abf262c16bfee5068`
- adapter_sha256_head: `a7bf45584aa7f06d048b2733532b4b6dc6c915c5f428ca1af48f2e125b023375`
- weights_different: `True`

Conclusion: adapter checkpoint is considered safe for eval only if `weights_different=True` and both weight files exist.
