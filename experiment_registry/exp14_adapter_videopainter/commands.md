# Commands

PAI clean worktree:

```bash
cd /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate
```

Gate2000 launcher used:

```bash
bash exp14_adapter_videopainter/scripts/launch_videopainter_adapter_gate2000_pai.sh
```

DAVIS eval launcher used:

```bash
CUDA_VISIBLE_DEVICES=0 \
NUM_INFERENCE_STEPS=50 \
NUM_FRAMES=49 \
bash exp14_adapter_videopainter/scripts/run_videopainter_adapter_davis_eval_pai.sh full
```

The eval script first supports debug mode:

```bash
CUDA_VISIBLE_DEVICES=0 \
DEBUG_LIMIT_VIDEOS=3 \
DEBUG_NUM_INFERENCE_STEPS=6 \
bash exp14_adapter_videopainter/scripts/run_videopainter_adapter_davis_eval_pai.sh debug
```

Do not rerun gate2000 unless a new adapter design is created. The current
gate2000 completed and produced a negative DAVIS50 result.
