# Exp15 DAVIS50 OR Benchmark

DAVIS50-only object-removal benchmark using DAVIS2017 foreground annotations.

This experiment is separate from the earlier OR150 plan. It does not use
YouTubeVOS100, does not start training, and does not use VBench.

Main protocol:

- input = original DAVIS2017 full-resolution frames;
- mask = DAVIS2017 foreground object annotation, nonzero means remove;
- output = each method's raw object-removal prediction;
- metric = background-region preservation, no hard comp before scoring.

Large generated frames/videos stay outside git under PAI `logs/target_eval`.
