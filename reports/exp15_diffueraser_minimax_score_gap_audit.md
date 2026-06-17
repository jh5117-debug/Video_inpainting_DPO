# Exp15 DiffuEraser / MiniMax Score Gap Audit

## Summary

The current Exp15 OR numbers should not be compared directly to MiniMax-Remover Table 2.

## Causes

1. **Dataset mismatch**
   - MiniMax paper: all 90 DAVIS videos and 200 Pexels videos.
   - Exp15: DAVIS50 subset only.

2. **Metric mismatch**
   - MiniMax TC follows COCOCO/AVID with CLIP-ViT-H/B-14 features.
   - Exp15 `TC_bg` is a simple pixel temporal-difference proxy.
   - Exp15 `SSIM_bg_ignore_mask` is a background-preservation proxy; the paper does not fully specify SSIM implementation in the PDF text.

3. **Mask mismatch risk**
   - Exp15 uses DAVIS2017 foreground annotations directly.
   - MiniMax paper uses GroundedSAM2 for Pexels and refers to DAVIS object masks, but exact DAVIS mask manifest/code must be verified.

4. **Inference mismatch**
   - MiniMax reports 480p, frame length 81, 6 sampling steps for its method.
   - Exp15 DiffuEraser wrappers use project OR settings, including 512x288 and existing DiffuEraser OR code. This is not MiniMax's baseline setup unless cross-checked against released scripts.

5. **Output / visualization bug risk**
   - The old visual MP4s used OpenCV `mp4v`, which can show green-cast playback in some environments.
   - Debugged raw frames and contact sheets are normal RGB, so the likely visual bug is encoding/player compatibility rather than all outputs being masks.

6. **MiniMax was not actually run**
   - Current Exp15 table has MiniMax blocked. Any MiniMax paper comparison is external context only, not a reproduced local result.

## Conclusion

Current Exp15 scores cannot be used as a direct MiniMax paper comparison. They are internal DAVIS50-subset OR diagnostics. A paper-aligned comparison requires DAVIS90, paper-compatible TC, verified mask manifests, and real MiniMax inference in its own environment.
