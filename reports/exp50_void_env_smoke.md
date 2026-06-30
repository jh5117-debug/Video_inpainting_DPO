# Exp50 VOID Environment Smoke

Status: `VOID_ENV_PARTIAL`.

- Env path: `/home/hj/conda_envs/void_exp50`
- Env type: Python venv with system site packages
- Full 5B model load: no
- Training/inference: no
- VOID repo path handling: appended after core imports to avoid shadowing HuggingFace `datasets`.

## Notes

Imports and CUDA smoke are usable, but exact official version pins are not fully satisfied. I did not reinstall heavyweight CUDA packages during smoke.

## Failed Imports

- none

## Version Pin Issues

- `torch`: expected=2.7.1; got=2.6.0+cu126
- `torchvision`: expected=0.22.1; got=0.21.0+cu126
- `diffusers`: expected=0.33.1; got=0.37.1
- `accelerate`: expected=1.12.0; got=1.13.0
- `transformers`: expected=4.57.1; got=5.5.4
- `safetensors`: expected=0.6.2; got=0.7.0
- `deepspeed`: expected=0.17.6; got=0.18.9
- `numpy`: expected=1.26.4; got=2.1.2

## CUDA

- `torch.cuda.is_available`: PASS True
- `torch.cuda.device_count`: PASS 8
- `bf16_support`: PASS True
- `small_matmul_backward`: PASS loss=59.136169
