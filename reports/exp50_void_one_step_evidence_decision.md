# Exp50 VOID One-Step Evidence Decision

Time: 2026-07-01T00:21:49.983380+00:00

## Status

- One-step video evidence: `VOID_ONE_STEP_PASS`
- 10-step micro gate: `VOID_ADAPTER_10STEP_NEGATIVE`
- VOID role after Exp50: `VOID_BASELINE_AND_LOSER_GENERATOR_READY`

## One-Step Evidence

H4b generated heldout4 Step0/Step1 raw videos from the saved one-step adapter and reviewed all evidence sheets. The one-step checkpoint reload was valid, heldout forward was finite, Step1 differed from Step0 by a small nonzero amount, and no collapse, systematic outside damage, or systematic tone drift was observed.

Mean H4b deltas:

- full PSNR: NA
- outside PSNR: NA
- mask PSNR: NA
- affected PSNR: NA
- boundary PSNR: NA
- visual better/tie/worse: 0/3/1

This upgrades the one-step gate from `VOID_ONE_STEP_PARETO_MIXED` to `VOID_ONE_STEP_PASS` because the previous blocker was missing heldout video evidence, not a failed optimization check.

## 10-Step Decision

The conditional 10-step micro gate ran exactly 10 optimizer steps on train4 and evaluated heldout4. It is negative, not blocked: the run finished technically, but does not meet promising/positive criteria.

Mean 10-step deltas:

- full PSNR: -0.0009652339261512211
- SSIM: -0.00234073665851095
- mask PSNR: -0.22987799889405647
- affected PSNR: 0.01934061651387431
- boundary PSNR: -0.06303431007626781
- outside PSNR: 0.043421852091388935
- visual better/tie/worse: 0/3/1

The gate fails because only one mean local/effect metric improved, 3/4 samples were metric-worse, and visual review showed no clear heldout improvement. There was no collapse or systematic outside failure, but there is also no adapter-positive evidence.

## Safety

No VOR-Eval, no hard comp, no long training, and no VOID official source edits were used.
