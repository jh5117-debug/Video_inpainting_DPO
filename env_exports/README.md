# VideoDPO Environment Exports

`videodpo_hal_diffueraser_compat.environment.yml` and
`videodpo_hal_diffueraser_compat.pip_freeze.txt` were exported on HAL after a
CPU-only VideoDPO smoke test passed with the existing `diffueraser` conda
environment.

This is a compatibility manifest for reproducing the current working HAL smoke,
not the clean official VideoDPO environment.  For a clean official-style env,
run:

```bash
CREATE_ENV=1 INSTALL_REQUIREMENTS=1 CONDA_ENV=videodpo \
bash DPO_finetune/scripts/videodpo_env_smoke_and_export.sh
```

For SC, if the existing DiffuEraser-DPO environment must be reused first, run:

```bash
CONDA_ENV=diffueraser \
bash DPO_finetune/scripts/videodpo_env_smoke_and_export.sh
```
