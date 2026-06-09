# Exp10 Status

status: blocked_by_external_sigterm

The previous PAI run reached partial Stage1, but continuation attempts were
externally terminated with `SIGTERM`. A fresh run with a new `RUN_VERSION`,
`RESUME_FROM_CHECKPOINT=none`, and no `POLICY_INIT_PATH` was also terminated:

```text
RUN_VERSION=20260609_145145_exp10_fresh_d3n16_val24
failure=Signal 15 (SIGTERM) received by PID 2698917
time=2026-06-09 14:55:32 CST
```

Conclusion: do not keep relaunching until the PAI SIGTERM sender/policy is
identified or disabled for this workload.
