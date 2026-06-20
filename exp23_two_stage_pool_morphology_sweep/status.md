TRAINER_WIRED_PHY_LAUNCH_PENDING

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
