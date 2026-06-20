# Exp20 Candidate Checkpoint Identity Audit

- Scope: equal-step 112-step checkpoints only; first-wave 30-minute P4 is intentionally excluded.
- Static audit confirms checkpoint directories and safetensors checksums. Runtime strict-load is verified by the evaluator/trial runner before each eval.

| candidate | seed | steps | config hash | unet sha256 | brushnet sha256 | last_weights |
|---|---:|---:|---|---|---|---|
| EQ_P0 | 20260619 | 112 | 1d8cd54758b73251 | `96954b66dd708d824a48f64c095285bb7d9533091d330feffec91b2f9fa124da` | `b0268b3620112267a033c6bac025ba1c87d36b9fb6824afd22897f061a345354` | `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp20_autoresearch_scale_adaptive_region_dpo/trials/EQ_P0_1d8cd54758b73251/last_weights` |
| EQ_P4 | 20260619 | 112 | edbea07bb785e769 | `94e7ac27db73e092b26806738a42483faa94f9fcc7125ce011a719310fc4497f` | `c58931ed2725061c6b9bae9d216d6e3d89586268df0e5469c4546780a4f2c9c2` | `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp20_autoresearch_scale_adaptive_region_dpo/trials/EQ_P4_edbea07bb785e769/last_weights` |
| EQ_BF07 | 20260619 | 112 | 2bc98e58514fb1da | `a360562c2bcc45adafc7c8ac2d9faf78f54da22a296d5c625217f14a9ffa605d` | `88172831112126bff526afbe88df2093fe49b5a6e97988a16cdef6155831c074` | `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp20_autoresearch_scale_adaptive_region_dpo/trials/EQ_BF07_2bc98e58514fb1da/last_weights` |
| EQ_AD04 | 20260619 | 112 | 77a0ed002ad3955d | `b1f2bc1d8837e6345e9921fa4a50573efcdc556af03ed56840129e371acabe1c` | `312da0b2f5d50ba2cf717af1416485e937db4cde6aee1b15dbfd0f90c4bb3fc7` | `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp20_autoresearch_scale_adaptive_region_dpo/trials/EQ_AD04_77a0ed002ad3955d/last_weights` |
