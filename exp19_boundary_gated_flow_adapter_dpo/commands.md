# Commands

Primary PAI launcher:

```bash
nohup bash exp19_boundary_gated_flow_adapter_dpo/scripts/launch_exp19_overnight_pai.sh \
  > logs/pipelines/exp19_boundary_gated_flow_adapter_dpo_overnight.log 2>&1 &
echo $! > logs/pipelines/exp19_boundary_gated_flow_adapter_dpo_overnight.pid
```

Local validation:

```bash
python -m py_compile exp19_boundary_gated_flow_adapter_dpo/code/*.py
bash -n exp19_boundary_gated_flow_adapter_dpo/scripts/*.sh
git diff --check
```
