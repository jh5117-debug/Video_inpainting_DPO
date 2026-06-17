# Commands

Prepare limit=100 propagation cache on PAI:

```bash
bash exp18_multiframe_propagation_gated_dpo/scripts/prepare_exp18_cache_limit100_pai.sh
```

Run Stage1-500 gates:

```bash
bash exp18_multiframe_propagation_gated_dpo/scripts/launch_exp18_stage1_gates_pai.sh
```

Run the whole guarded flow:

```bash
bash exp18_multiframe_propagation_gated_dpo/scripts/launch_exp18_overnight_pai.sh
```

Current HAL session note: PAI host `pai` is not resolvable and `/mnt/workspace`
is not mounted here, so the scripts are prepared but not launched from HAL.

