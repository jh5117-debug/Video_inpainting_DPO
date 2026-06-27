# Exp35 MiniMax 10-Step Forensic Audit

Status: `MINIMAX_NOCHANGE_CAUSE_UTILITY_SCALE_TOO_WEAK`

Linear utility stayed effectively constant near 0.5; gradients are too weak for visible output movement.

## Scope

- Source run: Exp30 MiniMax Gate64 adapter V3.
- Training performed in this milestone: false.
- Flow target: `epsilon - z0`.
- Exp30 trainable scope: all `Transformer3DModel` parameters.
- Bad-noise / hard-timestep miner used in Exp30: false.

## Checkpoint And Parameter Delta

### frozen

- Common checkpoint tensors: `461`.
- Missing / unexpected keys: `0` / `0`.
- Parameter count read: `1127055424`.
- Mean abs delta: `1.5329060227168864e-08`.
- Max abs delta: `8.106231689453125e-06`.
- Delta / param norm ratio: `5.6404525516172905e-06`.
- Zero-delta tensors: `60`.

### ema

- Common checkpoint tensors: `461`.
- Missing / unexpected keys: `0` / `0`.
- Parameter count read: `1127055424`.
- Mean abs delta: `1.5302821461092914e-08`.
- Max abs delta: `8.106231689453125e-06`.
- Delta / param norm ratio: `5.630459939756668e-06`.
- Zero-delta tensors: `60`.

## Output Diff

- Compared rows: `32`.
- Byte-identical rows: `0`.
- Mean full abs diff: `0.13143352206508793`.
- Mean mask abs diff: `0.18672874342540607`.
- Mean affected abs diff: `0.1731182035360047`.
- Mean outside abs diff: `0.10850902535158265`.
- Max abs diff: `28.0`.

Step10 is not byte-identical to Step0, so the checkpoint/inference path is not obviously falling back to Step0. The movement is, however, sub-perceptual and not quality-positive.

## Loss / Utility / Timestep Scale

### frozen

- Loss mean: `0.6931505858898163`.
- Linear utility mean/min/max: `0.4999982982873917` / `0.49997058510780334` / `0.5000085830688477`.
- Abs margin mean: `2.8578052297234536e-05`.
- Grad norm mean/max: `0.2986026735762829` / `1.0291674751425297`.
- t min/mean/max: `0.24` / `0.465` / `0.69`.

### ema

- Loss mean: `0.6931464791297912`.
- Linear utility mean/min/max: `0.5000003516674042` / `0.49999284744262695` / `0.5000050663948059`.
- Abs margin mean: `1.2780050747096539e-05`.
- Grad norm mean/max: `0.30134333519000756` / `1.0291811038318481`.
- t min/mean/max: `0.24` / `0.465` / `0.69`.

## Conclusion

Root-cause status for this milestone: `MINIMAX_NOCHANGE_CAUSE_UTILITY_SCALE_TOO_WEAK`. The strongest evidence is that Exp30 had valid data and nonzero checkpoint/output movement, but the utility stayed near constant and parameter/output movement was too small to matter. Next step is inference sensitivity positive-control before changing recipes.
