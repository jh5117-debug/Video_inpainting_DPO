# Status

`COMPLETED_NEGATIVE_STAGE1_GATES`

Exp17 Stage1-1000 gates completed on PAI for:

- `exp17a_positive_s1_1000`
- `exp17b_saturation_s1_1000`
- `exp17c_combined_s1_1000`

All variants were evaluated on DAVIS10 against SFT-48000 and Exp11 outer b0.75
S2. The best Exp17 variant was `exp17b_saturation_s1_1000`, but it did not beat
Exp11 on primary metrics or visual inspection.

No Stage1-2000 extension was launched.
No Stage2 was launched.
No VBench was used.

Decision:

```text
Stop Exp17 as a negative ablation for now.
Current best remains Exp11 outer b0.75 S2.
```
