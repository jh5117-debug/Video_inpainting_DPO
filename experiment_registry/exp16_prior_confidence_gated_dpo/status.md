# Status

`STAGE1_500_LIMIT100_DAVIS10_SANITY_COMPLETED`

Completed on PAI:

- limit=100 real ProPainter prior cache;
- confidence audit;
- Stage1 preflight;
- Stage1 500 small gate;
- checkpoint-250, checkpoint-500, `last_weights`, and dpo_diag.
- Exp16 DPO-S1 + SFT-S2 hybrid checkpoint for eval loading;
- DAVIS10 metric / visual sanity;
- confidence diagnostic fix and offline confidence summary.

Not launched:

- Stage2;
- full prior cache;
- full 2000+2000 training;
- DAVIS50/YouTubeVOS100 full eval.

Decision:

```text
Do not launch full prior cache or Stage1 2000 yet.
```

Reason: DAVIS10 shows Exp16 is better than SFT-48000 but still below Exp11
outer b0.75 S2 on the main metrics, with more negative visual cases than
positive ones.
