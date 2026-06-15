# VideoPainter Adapter Gate2000 Final Report

Date: 2026-06-16 CST

## Status

blocked_missing_weights_and_pai_hf_network_unreachable

No trainer preflight was run. No gate2000 training was launched. No dpo_diag, checkpoint, DAVIS eval, or visualization was produced.

## Exact Blocker

The required weights are not present:

- `third_party/VideoPainter/ckpt/CogVideoX-5b-I2V`
- `third_party/VideoPainter/ckpt/VideoPainter/checkpoints/branch`

PAI cannot download them from Hugging Face because network access to HF fails with `Network is unreachable`.

## Conclusion

The Exp14 code path is ready, but the experiment cannot proceed until the official VideoPainter / CogVideoX checkpoints are provided on PAI.
