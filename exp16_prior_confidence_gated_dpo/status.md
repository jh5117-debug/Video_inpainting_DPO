# Status

Status: `BLOCKED_PENDING_REAL_PRIOR_CACHE_AND_FULL_X0_TRAINER_INTEGRATION`

What is ready:

- isolated Exp16 folder and registry;
- prior manifest/cache builder;
- confidence-map and latent-x0 prior loss helpers;
- preflight script that blocks without real ProPainter prior paths;
- launcher that does not start old Exp11 loss as Exp16.

What is not ready:

- full Stage1/Stage2 trainer integration for `L_prior`, `L_gen`, and
  `L_boundary_extra`;
- full ProPainter prior cache for the training manifest;
- PAI-side preflight with real model tensors.

No Exp16 training has been launched.

