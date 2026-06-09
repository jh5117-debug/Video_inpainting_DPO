# Exp10 Commands

Exp10 is prepared but not launched by default. After the 2026-06-09 PAI
`SIGTERM` continuation failures, the next retry must be fresh, not resumed from
the interrupted checkpoint.

Fresh PAI run:

```bash
bash scripts/pai_launch_exp10_fresh_gpus0_6.sh
```

Manual run:

```bash
cd /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO
RUN_EXPERIMENTS=exp10 bash scripts/launch_exp09_10_11_pai.sh
```

Sequential run after Exp9 only when explicitly requested:

```bash
RUN_EXPERIMENTS=exp9,exp10 bash scripts/launch_exp09_10_11_pai.sh
```
