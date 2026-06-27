# Exp36 MiniMax No-Change Forensic Audit

Status: `MINIMAX_NOCHANGE_CAUSE_UTILITY_SCALE_TOO_WEAK`

This milestone performed no new training and no new inference. It reread
Exp30 and Exp35 checkpoints, metrics, diagnostics, and visual-review reports
to classify why MiniMax remains plumbing-positive but not quality-positive.

## A1 Parameter Update

Exp30 frozen/EMA 10-step recipes updated the MiniMax transformer checkpoint,
but only at a very small scale:

- Frozen delta / parameter norm: `5.6404525516172905e-06`
- EMA delta / parameter norm: `5.630459939756668e-06`
- Mean absolute delta: about `1.53e-08`
- Max absolute delta: `8.106231689453125e-06`
- Zero-delta tensors: `60` of `461`

Exp35 later proved larger updates are possible:

- Winner-SFT LR `1e-5`: step10 delta probe `1.4444718289041703e-05`
- Winner-SFT LR `3e-5`: step10 delta probe `4.1587465275938484e-05`
- Winner-SFT LR `1e-4`: step10 delta probe `0.0002197052692736179`

However, those larger updates hurt heldout quality. Exp35 R1/R2/R3 rescue
recipes had nonzero deltas around `9.7e-06`, but heldout local and boundary
metrics were negative.

## A2 Checkpoint And Inference Identity

Checkpoint identity is not the dominant failure mode:

- Exp30 checkpoint tensor keys matched: `461/461`.
- Missing/unexpected keys: `0/0`.
- Step10 outputs were not byte-identical to Step0.
- Exp35 inference-sensitivity positive-control showed the inference path
  responds to MiniMax transformer weight perturbations.

Current classification:

- `MINIMAX_NOCHANGE_CAUSE_CHECKPOINT_LOAD`: not supported.
- `MINIMAX_NOCHANGE_CAUSE_TRAINABLE_SCOPE`: not supported by current evidence.

## A3 Output Sensitivity

Exp30 Step0 vs Step10 output diff:

- Rows compared: `32`
- Byte-identical rows: `0`
- Mean full/mask/affected/outside absolute pixel diff:
  `0.13143352206508793`, `0.18672874342540607`,
  `0.1731182035360047`, `0.10850902535158265`
- Max absolute pixel diff: `28.0`

Exp35 R1/R2/R3 output movement was stronger in review frames, but still not
quality-positive:

- R1 mean review-frame pixel diff: `3.355972468852997`
- R2 mean review-frame pixel diff: `3.3715626001358032`
- R3 mean review-frame pixel diff: `3.54348715643088`

This rules out exact no-op behavior but confirms low or harmful useful
sensitivity.

## A4 Loss / Utility Scale

Exp30 Linear-DPO utilities stayed essentially constant:

- Frozen utility mean/min/max:
  `0.4999982982873917` / `0.49997058510780334` /
  `0.5000085830688477`
- Frozen abs margin mean: `2.8578052297234536e-05`
- EMA utility mean/min/max:
  `0.5000003516674042` / `0.49999284744262695` /
  `0.5000050663948059`
- EMA abs margin mean: `1.2780050747096539e-05`

Exp35 hard-noise rescue increased some gradients and pixel movement but did
not convert that movement into heldout repair. R1/R2/R3 all had negative mean
mask, boundary, and outside PSNR deltas and `0` quality-positive visual rows.

## A5 Timestep / Noise

Exp30 did not use bad-noise / hard-timestep mining. Its t range was `0.24` to
`0.69`, mean `0.465`. Exp35 mined hard states and used fixed `hard_state_A`,
but R1/R2/R3 still failed. Therefore "timestep too easy" may be a contributing
factor for Exp30, but it is not sufficient to explain Exp35's rescue failure.

## Conclusion

Root-cause status: `MINIMAX_NOCHANGE_CAUSE_UTILITY_SCALE_TOO_WEAK`.

More precisely: MiniMax updates and inference are real, but the previous
preference objectives either produce margins too close to zero or push outputs
in directions that damage mask/boundary/outside quality. Exp36 should next
rerun a bounded inference-sensitivity check under the new branch, then test
whether a winner-preserving / local-region objective can create useful
movement before any 30-step unlock.

No 30-step, long training, RC-FPO, protected-lane action, or universal-adapter
claim is unlocked by this audit.

