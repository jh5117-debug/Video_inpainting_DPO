BLOCKED_GPU7_GHOST_AFTER_REAL_PHY_LAUNCH

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

2026-06-21 PAI launch result:

- PAI HEAD: `d9d7077c281af33e7186f890d5175e4d470c1d8b`.
- Controller PID `1285825`, torchrun PID `1285828`, rank PIDs `1285905-1285908` all used `/mnt/nas/hj/conda_envs/diffueraser/bin/Phy`.
- Fresh Exp11 twin Stage1 started and wrote step-1 DPO diagnostics.
- Rank3 failed with CUDA OOM because GPU7 still has PID `1758887` `[Not Found]` using `58060 MiB`.
- Queue state: `FAILED`, model `fresh_exp11_outer_b075`, pair `phaseA_scale1_pair001_outer2`.
- No Exp23 worker remains alive.
