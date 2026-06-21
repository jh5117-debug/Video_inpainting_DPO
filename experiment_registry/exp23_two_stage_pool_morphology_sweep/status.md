RUNNING_ON_GPU2_4_5_6

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

2026-06-21 GPU2/4/5/6 retry:

- Pair: `phaseA_scale1_pair001_outer2_gpus2456`.
- Controller PID: `1289732`.
- Torchrun PID: `1289735`.
- Rank PIDs: `1289812`, `1289813`, `1289814`, `1289815`.
- GPUs: `2,4,5,6`.
- Current model: `fresh_exp11_outer_b075`.
- 15-minute monitor passed; Stage1 reached at least step 170 with finite loss and gradients.
- Background monitor PID: `1291494`; log: `exp23_two_stage_pool_morphology_sweep/runtime/monitor_gpus2456.log`.
- 500-step monitor passed at 2026-06-21 08:52 CST.
- `checkpoint-500` exists under the fresh Exp11 Stage1 output directory.
- Training continued past step 510 with no OOM, Traceback, or NaN grep hits.

2026-06-21 pair completion:

- Pair `phaseA_scale1_pair001_outer2_gpus2456` completed on GPU2/4/5/6.
- Queue status: `STAGE1_STAGE2_PAIR_COMPLETED`.
- Fresh Exp11 twin: Stage1 2000 + Stage2 2000 completed.
- Candidate `candidate_scale1_outer2_b075`: Stage1 2000 + Stage2 2000 completed.
- Candidate Stage2 final checkpoint and `last_weights` exist under:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp23_two_stage_pool_morphology_sweep/pairs/phaseA_scale1_pair001_outer2_gpus2456/candidate_scale1_outer2_b075/stage2/`
- Final GPU2/4/5/6 memory after completion: `0 / 244 / 4 / 292 MiB`.
- No Exp23 `Phy` process remains.
- DAVIS50 evaluation has not run yet and remains the next required gate.
