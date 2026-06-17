# Status

Status: `STAGE1_500_LIMIT100_COMPLETED`

What is ready:

- isolated Exp16 folder and registry;
- prior manifest/cache builder;
- confidence-map and latent-x0 prior loss helpers;
- preflight script that blocks without real ProPainter prior paths;
- launcher that does not start old Exp11 loss as Exp16.

Completed:

- real ProPainter prior cache, limit=100;
- confidence audit on the cache;
- Stage1 preflight with real prior frames and latent-x0 target losses;
- Stage1 500 small gate;
- checkpoint-250, checkpoint-500, last_weights, and dpo_diag.

What is not ready:

- Stage2 trainer integration for `L_prior`, `L_gen`, and `L_boundary_extra`;
- full ProPainter prior cache for the training manifest;
- full 2000+2000 training;
- DAVIS/YouTubeVOS metric and visual evidence.

Stage1 500 was launched and completed. Stage2/full training was not launched.
