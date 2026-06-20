TRAINER_WIRED_PHY_LAUNCH_PENDING

2026-06-21:

- User authorized force release of GPU4-7.
- GPU4-6 high-expert multigen workers were terminated with targeted TERM.
- GPU4-6 are now idle-level.
- GPU7 still has a 58060 MiB `[Not Found]` NVML allocation with no visible `/proc` holder.
- Exp23 training was not launched at that time because the branch still lacked real Stage1/Stage2 trainer and queue/evaluator plumbing.

2026-06-21 trainer wiring update:

- Isolated Stage1 and Stage2 trainers are now present under Exp23.
- A first paired Phase A runner is now present.
- The PAI launcher creates/uses `$CONDA_PREFIX/bin/Phy` and starts torch distributed with `--nproc_per_node=4`.
- Local py_compile, unit tests, shell syntax check, and diff whitespace check passed.
- Next state depends on the PAI relaunch attempt; GPU7 remains a likely CUDA blocker.
