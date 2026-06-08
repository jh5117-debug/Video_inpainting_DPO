# Exp10 Commands

Exp10 is prepared but not launched by default.

```bash
cd /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO
RUN_EXPERIMENTS=exp10 bash scripts/launch_exp09_10_11_pai.sh
```

Sequential run after Exp9 only when explicitly requested:

```bash
RUN_EXPERIMENTS=exp9,exp10 bash scripts/launch_exp09_10_11_pai.sh
```
