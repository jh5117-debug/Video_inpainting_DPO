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

If the smoke reports missing modules such as `pytorch_lightning`, rerun with:

```bash
CONDA_ENV=diffueraser INSTALL_MINIMAL=1 \
bash DPO_finetune/scripts/videodpo_env_smoke_and_export.sh
```

`INSTALL_MINIMAL=1` installs only VideoDPO smoke/runtime dependencies listed in
`videodpo_requirements.minimal_no_torch.txt`; it intentionally excludes
`torch`, `torchvision`, `numpy`, and `xformers`.  It does include
`setuptools<81` because `pytorch_lightning==1.9.x` imports `pkg_resources`.
