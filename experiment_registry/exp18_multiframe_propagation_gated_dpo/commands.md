# Commands

```bash
bash exp18_multiframe_propagation_gated_dpo/scripts/launch_exp18_overnight_pai.sh
```

PAI command actually used:

```bash
nohup env ROOT=/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp18_gate CUDA_VISIBLE_DEVICES=0 RUN_VERSION=20260618_exp18_gate \
  bash exp18_multiframe_propagation_gated_dpo/scripts/launch_exp18_overnight_pai.sh \
  > logs/pipelines/exp18_multiframe_propagation_gated_dpo_driver.log 2>&1 &
```

Additional true DAVIS10 hybrid eval:

```bash
bash exp18_multiframe_propagation_gated_dpo/scripts/run_exp18_davis10_hybrid_eval_pai.sh
```
