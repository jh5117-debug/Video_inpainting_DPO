BLOCKED_TRAINER_AND_GPU7_GHOST

2026-06-21:

- User authorized force release of GPU4-7.
- GPU4-6 high-expert multigen workers were terminated with targeted TERM.
- GPU4-6 are now idle-level.
- GPU7 still has a 58060 MiB `[Not Found]` NVML allocation with no visible `/proc` holder.
- Exp23 training was not launched because the branch still lacks real Stage1/Stage2 trainer and queue/evaluator plumbing.
