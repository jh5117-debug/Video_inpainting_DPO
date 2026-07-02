# Exp59 VOID Kubric Gate8 Inference Status

Current status: `EXP59_OFFICIAL_INFERENCE_PROTOCOL_READY`

Milestone A read back Exp58B Kubric Gate8 and audited the native inputs on PAI. The manifest has 8 rows, all required files exist, all videos decode, all clips are 24 frames at 128x128 and 8 fps, and every quadmask contains `0|63|127|255`.

The dataset is sufficient for official VOID inference diagnostics but weak for adapter training because all metadata rows have `target_hit=false`.

No training, preference forward, zero-gap, one-step, or 10-step has been run in Exp59.

Milestone B audited official VOID pass1 inference. The official script accepts the 128x128 Kubric inputs through its normal preprocessing path by resizing video and quadmask tensors to `config.data.sample_size`, which defaults to `384x672`. This is sufficient to continue to input materialization, with the caveat that metrics must compare at a common resolution.
