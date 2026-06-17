# Exp15 MiniMax-Remover Env Build Report

Current status: `BLOCKED_DEPENDENCY` / isolated env not yet built.

## PAI Evidence

Checked paths:

```text
/mnt/nas/hj/official_repos/MiniMax-Remover_28e12b4                 exists
/mnt/nas/hj/H20_Video_inpainting_DPO/weights/minimax_remover/current exists
/mnt/nas/hj/conda_envs/minimax_remover                             missing
/mnt/nas/hj/conda_envs/minimax_remover/bin/python                  missing
```

MiniMax official `requirements.txt` includes:

```text
torch==2.7.1
torchvision==0.22.1
diffusers==0.33.1
accelerate==0.30.1
decord==0.6
moviepy==1.0.3
...
```

The shared DiffuEraser env must not be used because it lacks newer Wan/diffusers components such as `AutoencoderKLWan` / `FP32LayerNorm`.

## Conclusion

MiniMax was not run. No MiniMax local score should be reported. Building this env is a separate dependency gate because it requires a large torch/diffusers stack and should be isolated from the current DiffuEraser training env.

## Safe Next Action

Create:

```text
/mnt/nas/hj/conda_envs/minimax_remover
```

then install MiniMax official requirements and validate imports before any DAVIS50 OR inference.
