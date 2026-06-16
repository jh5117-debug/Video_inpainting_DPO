# VideoPainter Adapter Eval Precheck

- project_root: `/mnt/nas/hj/H20_Video_inpainting_DPO_exp14_videopainter_gate`
- videopainter_root: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter`
- base_model: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter/ckpt/CogVideoX-5b-I2V`
- baseline_branch: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter/ckpt/VideoPainter/checkpoints/branch`
- adapter_checkpoint: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/exp14_adapter_videopainter/runs/gate2000/last_weights`
- output_dir: `/mnt/nas/hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/logs/target_eval/exp14_videopainter_adapter_gate2000_davis`
- DAVIS root: `/mnt/workspace/hj/nas_hj/data/external/davis_432_240`
- DAVIS available videos: 50
- selected videos: 50
- debug: False
- num_frames: 49
- num_inference_steps: 50
- hard comp: yes
- mask dilation: no
- Gaussian blur: no
- VBench: no
- metric backend: `inference/metrics.py`
