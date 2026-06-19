# Exp20 Safe Autoresearch Program

This experiment borrows only the safe parts of autoresearch and AI-Scientist-v2:

- fixed train wall-clock per fast trial;
- immutable evaluator and immutable dev split;
- config-only children after parity;
- best-first expansion of parent/child experiment nodes;
- crash nodes are debugged once, then marked crash/blocked;
- every trial has a config hash and append-only result row.

It explicitly does not use self-modifying Python during the sweep. Python code is
changed only before parity and committed for review. Trials modify YAML/JSON
configuration only.

Current source-of-truth baseline:

- Exp11 boundary outer b0.75 S2;
- mask weight 1.0, outer boundary 0.75, outside 0.05;
- log-ratio normalized gap, clipped loser, winner anchor;
- DAVIS50/YouTubeVOS100 raw6 hard-comp, no PCM, no dilation, no blur.
