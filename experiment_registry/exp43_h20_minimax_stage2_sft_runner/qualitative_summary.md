# Exp43 Qualitative Summary

No Exp43 output videos have been generated yet.

## 2026-06-29 H20 GPU Release

No qualitative model review was performed. This milestone only confirms that H20
GPUs and CUDA are available for the next gates.
# Exp43 Qualitative Summary

Current status: `H20_EXP43_SFT_BLOCKED`.

## 2026-06-29 SFT-A 30-Step Cell

The run generated raw outputs, side-by-side videos, 16-frame strips, and
midframe sheets for `search24` and `shadow24`. Visual review remains pending,
and no PASS/POSITIVE claim is allowed.

The numeric gate already failed strongly on search and shadow, so no longer
SFT/DPO/500-step stage was unlocked. Video evidence should still be inspected
before making any qualitative statement beyond `metric-negative / review
pending`.

No Exp43 output videos exist yet. Prior Exp41 visual/protocol review found no
mask reversal, hidden comp, or GT leakage, but did not establish MiniMax quality
positive evidence.

Any future Exp43 PASS/POSITIVE/THIRD_BACKBONE status requires raw videos,
temporal strips, region crops, per-video metrics, aggregate metrics, visual
review CSV, and direct review of all search/shadow evidence.

BF16 preflight produced no quality videos and makes no visual-quality claim.

Data readiness decoded required frame directories only. It does not review new
MiniMax trained outputs and makes no visual-quality claim.
