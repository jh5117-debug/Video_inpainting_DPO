# Exp38 MiniMax Failure Taxonomy

Date: 2026-06-28

Status: `MINIMAX_FAILURE_TAXONOMY_BUILT`

This milestone converts the Exp30/35/36/37 MiniMax evidence into an executable
decision tree. It did not launch training, inference, GPU work, or protected
lane actions.

## Evidence Base

- Exp30 Gate64 MiniMax frozen/EMA 10-step:
  `MINIMAX_ADAPTER_RECIPE_NOT_READY`, visual better `0/32`.
- Exp35 hard-noise rescue:
  `MINIMAX_RESCUE_RECIPE_NOT_READY`, visual better `0/48`.
- Exp36 sensitivity:
  `MINIMAX_INFERENCE_SENSITIVITY_PASS`, identity replay max MAE `0.0`,
  1.01x perturbation mean full/mask MAE `0.088218` / `0.156302`.
- Exp36 winner-SFT:
  train loss decreased and outputs moved, but visual better `0/24`.
- Exp37 LocalDPO-badnoise:
  R1 mixed metrics, R2/R3 negative metrics, each recipe visual better `1/16`,
  so the result is `MINIMAX_LOCALDPO_BADNOISE_PARETO_MIXED`.

## Taxonomy

| ID | Category | Current assessment | Evidence for | Evidence against | Remaining test | Next action | Stop condition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | Code/loading failure | mostly ruled out | Early no-change could have looked like fallback | Exp30/35 strict keys matched; Exp36 perturbation changed rendered outputs; checkpoints were loaded and reference delta stayed controlled | Official reproduction audit can still find protocol mismatch, not load failure | Audit official MiniMax protocol and checkpoint identity before new training | Any mismatch showing Step10/perturbed checkpoint not loaded stops DPO rescue until fixed |
| B | Adapter ignored by inference | ruled out | Old Step10 changes were visually tiny | Exp36 identity/perturbation test proves inference responds to trained transformer weights | Repeat sensitivity only after new scope if S1/S2 adapters are used | Include adapter-scale/roundtrip check for any new scope | If new scope is ignored, stop training and fix loader |
| C | Trainable scope too weak | unresolved but not primary | Exp36 S1 LoRA was structurally ready and may be too weak; full model updates may be diffuse | Exp30/35 used full transformer scope, so tiny LoRA is not the only cause | Scope v2 and train-overfit tests | Test scope only after positive-control and protocol audit | If train videos still do not improve under stronger scope, shift to objective/data |
| D | LR/update scale too weak | partially supported | Exp30 utility near 0.5, Step10 movement sub-perceptual; Exp35 classified weak utility scale | Exp36 high LR moved outputs and created artifacts; Exp37 R1 moved metrics but missed visual gate | Train-overfit ladder with bounded LR/scope | Calibrate on train-overfit before heldout DPO | If train does not visibly improve, do not proceed to heldout DPO |
| E | Objective signal too weak | strongly supported | Exp37 train-vs-heldout diagnosed `MINIMAX_OBJECTIVE_SIGNAL_TOO_WEAK`; R1 only 1/16 visually better | R1 had some positive mask/full metric movement, so objective is not entirely inert | Train-overfit diagnosis, SFT warmup ladder, local/effect metric targeting | Design staged objective with stronger local residual and winner anchor | If objective cannot improve train videos, stop DPO recipes |
| F | Bad-noise/timestep not aligned | unresolved | Exp37 hard states were outside-sane but lower-gradient than random baseline; dominant t=0.25 may be narrow | Bad-noise scan produced valid states and R1 got some local metric signal | Bad-noise v2 gradient-strength scan | Verify hard-state local gradient >= random median target before using as core recipe | If hard states are not stronger/useful, do not center recipe on bad-noise |
| G | Data too diverse/noisy | plausible | MiniMax official candidates had low yield; heldout often ties; broad Gate64 may mix defect families | Exp37 LocalDPO pool was clean and reviewed 48/48, but still not enough | LocalDPO v2 with balanced stronger defects if justified | Build stronger pool only after train-overfit/protocol audit | If cleaner data still cannot improve train videos, data is not sufficient |
| H | LocalDPO corruption too weak | plausible | Exp37 R1 made only one visible heldout improvement; many rows tied | Pool contained medium-hard/hard-plausible rows, no global collapse | Pool v2 with stronger bounded local/effect profiles | Increase local/effect difficulty while preserving outside | If stronger pool becomes trivial-bad/outside-damaging, reject it |
| I | Generalization failure | not primary yet | Heldout does not improve | Exp37 train-vs-heldout found train does not meaningfully improve either | Evaluate existing best checkpoints on train16/train32 and heldout16 | Diagnose train vs heldout before new objectives | If train improves but heldout fails, then scale/diversify data |
| J | Evaluation too insensitive | partly possible but not enough | Some metric deltas are tiny and visual ties dominate | Exp36 perturbation and Exp37 R1 produce measurable pixel/metric movement; visual review catches artifacts | Add train-overfit visual review and local/effect residual metrics | Keep metrics plus per-video visual review; do not rely on whole PSNR | If only invisible metric gains occur, no quality-positive promotion |

## Decision Tree

1. Confirm protected-lane and permission state before any GPU work.
2. If checkpoint/protocol audit finds mismatch, fix the isolated Exp38 wrapper
   and rerun smoke before any training.
3. If inference ignores a new adapter scope, fix checkpoint loading and stop
   training.
4. If MiniMax cannot visibly improve training rows under a supervised
   local-winner positive-control, do not run heldout DPO; expand scope/update
   calibration first.
5. If train improves but heldout fails, treat the problem as data diversity or
   generalization and build a stronger LocalDPO v2 pool.
6. If hard-noise v2 does not produce stronger useful local gradients than
   random states, do not center the objective on bad-noise.
7. Only after protocol, train-overfit, data, and hard-state gates pass may a
   preregistered 10-step recipe run.
8. 30-step remains locked until the 10-step gate is visibly positive.

## Immediate Next Action

Run Milestone B train-overfit diagnosis using existing checkpoints and pools.
The key question is whether MiniMax can visibly improve training videos at all.

No Exp38 result currently supports MiniMax third-backbone success or universal
adapter language.

