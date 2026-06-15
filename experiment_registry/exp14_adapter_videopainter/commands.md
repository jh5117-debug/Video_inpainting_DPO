# Commands

PAI smoke guards:

```bash
cd /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO || exit 1
git pull --ff-only
bash exp14_adapter_videopainter/scripts/run_videopainter_adapter_smoke1_pai.sh
```

The current expected result is a safe `BLOCKED` message until the adapter
trainer is implemented.

