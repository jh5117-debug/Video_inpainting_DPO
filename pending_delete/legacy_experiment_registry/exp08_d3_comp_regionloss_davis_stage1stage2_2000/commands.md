# Commands

PAI launch is manual-only from the IDE agent. Use:

```bash
bash scripts/launch_exp8_d3_comp_regionloss_s1s2_2000_davis_pai.sh
```

Expected wrapper log:

```text
logs/pipelines/exp08_d3_comp_regionloss_wingap_lose025_s1s2_2000_davis_pai.log
```

Expected PID file when launched with `nohup`:

```text
logs/pipelines/exp08_d3_comp_regionloss_wingap_lose025_s1s2_2000_davis_pai.pid
```

The script must stop before training if region loss, ProPainter prior, SFT-48000 weights, DAVIS data, or the D3 comp manifest are not valid.
