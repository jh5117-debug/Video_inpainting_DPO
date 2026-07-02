# Exp59 VOID Kubric Gate8 Inference Status

Current status: `VOID_KUBRIC_OFFICIAL_INFERENCE_PASS`

Milestone A read back Exp58B Kubric Gate8 and audited the native inputs on PAI. The manifest has 8 rows, all required files exist, all videos decode, all clips are 24 frames at 128x128 and 8 fps, and every quadmask contains `0|63|127|255`.

The dataset is sufficient for official VOID inference diagnostics but weak for adapter training because all metadata rows have `target_hit=false`.

No training, preference forward, zero-gap, one-step, or 10-step has been run in Exp59.

Milestone B audited official VOID pass1 inference. The official script accepts the 128x128 Kubric inputs through its normal preprocessing path by resizing video and quadmask tensors to `config.data.sample_size`, which defaults to `384x672`. This is sufficient to continue to input materialization, with the caveat that metrics must compare at a common resolution.

Milestone C materialized 8 official input folders under the writable PAI/NAS runtime root. The requested `/experiments/dpo/exp59...` output root is not writable by `hj`, so official inference outputs must use the writable log/runtime fallback.

Milestone D ran official VOID pass1 inference on exactly 8 Kubric Gate8 samples. The first attempt exposed a system `/usr/bin/ffmpeg` missing-`libblas.so.3` blocker; a controlled run-local `imageio-ffmpeg` shim fixed decode without modifying the base environment. Final outputs: 8/8 raw pass1 mp4, 8/8 tuple mp4, and 8/8 evidence packs.
