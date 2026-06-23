# Exp14 Commands

The user requested skipping smoke and attempting gate2000 directly. Use the
following command on PAI only after syncing this repo. The script still blocks
if the isolated adapter trainer is missing.

## PAI Gate2000 Precheck / Launch Guard

```bash
cd /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO || exit 1
git pull --ff-only
bash exp14_adapter_videopainter/scripts/launch_videopainter_adapter_gate2000_pai.sh
```

Expected current result:

```text
BLOCKED: isolated VideoPainter DPO adapter trainer is not implemented
```
