# Exp14 Commands

This HAL session did not run PAI smoke. Use the following commands on PAI only
after syncing this repo.

## PAI Precheck / Smoke1 Guard

```bash
cd /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO || exit 1
git pull --ff-only
bash exp14_adapter_videopainter/scripts/run_videopainter_adapter_smoke1_pai.sh
```

Expected current result:

```text
BLOCKED: adapter train script not implemented yet.
```

## PAI Smoke20 Guard

Only run after Smoke1 is implemented and passes.

```bash
cd /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO || exit 1
git pull --ff-only
bash exp14_adapter_videopainter/scripts/run_videopainter_adapter_smoke20_pai.sh
```

## Gate2000

Do not run until Smoke1 and Smoke20 pass and the user explicitly confirms.

