RUNNING_ON_GPU2_4_5_6

Core morphology implementation and initial tests are present.

2026-06-21 force-release result:

- GPU4 released: 141333 MiB -> 244 MiB.
- GPU5 released: 141045 MiB -> 4 MiB.
- GPU6 released: 142321 MiB -> 292 MiB.
- GPU7 remains occupied by NVML PID 1758887 `[Not Found]` at 58060 MiB with no visible `/proc` entry.

Full training had not started at the first force-release checkpoint because:

1. GPU7 is still not fully released, and reset is prohibited.
2. The current Exp23 branch does not yet include a real Stage1/Stage2 trainer, paired queue/controller, or DAVIS50 evaluator.

2026-06-21 trainer wiring update:

- Isolated Stage1 and Stage2 trainers are now present under Exp23.
- A Phy torch.distributed runner now launches the first Phase A pair.
- Local code/test gates pass.
- PAI launch is pending. GPU7 remains a likely CUDA blocker due its no-proc NVML allocation.

2026-06-21 PAI launch result:

- Real Phy torch.distributed launch succeeded.
- Fresh Exp11 twin Stage1 completed optimizer step 1 and wrote diagnostics.
- Rank3 on GPU7 failed with CUDA OOM because PID `1758887` `[Not Found]` still retains `58060 MiB`.
- Queue state is `FAILED`; no Exp23 Phy worker remains alive.

2026-06-21 GPU2/4/5/6 retry:

- Restarted as `phaseA_scale1_pair001_outer2_gpus2456`.
- Controller PID `1289732`; rank PIDs `1289812-1289815`.
- 1-minute and 15-minute monitors passed.
- Fresh Exp11 Stage1 is running; step >= 170 at 15-minute check.
- Background monitor PID `1291494` is sampling every 5 minutes.
