# Exp11 Flow-Prior Consistency DPO

Experiment name: `exp11_flow_prior_consistency_dpo_s1s2_2000_davis_pai`

Purpose: build on Exp10 and add flow / ProPainter prior / boundary consistency to address purple haze, patches, boundary discontinuity, and temporal flicker.

Current status: blocked until the train-time flow/prior implementation audit passes.

The current repository supports DAVIS inference with ProPainter prior, but the DPO training loop does not yet expose a safe differentiable flow tensor or image/x0-space ProPainter-prior target. The master pipeline will write `reports/exp11_flow_prior_implementation_audit.md` and stop before training unless this audit is resolved.
