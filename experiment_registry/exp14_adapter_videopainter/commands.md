# Commands

PAI gate2000 precheck / launch guard:

```bash
cd /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO || exit 1
git pull --ff-only
bash exp14_adapter_videopainter/scripts/launch_videopainter_adapter_gate2000_pai.sh
```

The current expected result is a safe `BLOCKED` message until the isolated
adapter trainer is implemented.
