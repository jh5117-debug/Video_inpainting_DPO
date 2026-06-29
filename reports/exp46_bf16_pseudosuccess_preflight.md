# Exp46 Pseudo-Success BF16 Preflight

Status: EXP46_BF16_SAFE_READY

Output root: /home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp46_h20_minimax_pseudosuccess_sft/bf16_preflight_20260630_005259

P0-P7 were run on H20 pseudo-success runner manifests. No optimizer step was executed; P5 checkpoint save/reload was a dry-run only.

| case | rank | world size | status | loss | grad norm | checkpoint | peak bytes |
| --- | ---: | ---: | --- | ---: | ---: | --- | ---: |
| P0 | 0 | 1 | PASS | 2050.82861328125 | 2.831369161605835 |  | 134218752 |
| P1 | 0 | 1 | PASS |  |  |  | 6564385792 |
| P2 | 0 | 1 | PASS | 0.1915583312511444 | 0.0 | not_requested | 7628011008 |
| P3 | 0 | 1 | PASS | 0.1915583312511444 | 0.0 | not_requested | 7628011008 |
| P4 | 0 | 1 | PASS | 0.1915583312511444 | 2.066658728161611 | not_requested | 62842241024 |
| P5 | 0 | 1 | PASS | 0.1915583312511444 | 2.066658728161611 | PASS | 62842241024 |
| P6 | 0 | 2 | PASS | 0.1915583312511444 | 2.071865093266925 | not_requested | 65102424576 |
| P6 | 1 | 2 | PASS | 0.1976272612810135 | 2.071865093266925 | not_requested | 65102424576 |
| P7 | 0 | 8 | PASS | 0.1915583312511444 | 2.0687390765518066 | not_requested | 65102424576 |
| P7 | 1 | 8 | PASS | 0.1976272612810135 | 2.0687390765518066 | not_requested | 65102424576 |
| P7 | 2 | 8 | PASS | 0.19424597918987274 | 2.0687390765518066 | not_requested | 65102424576 |
| P7 | 3 | 8 | PASS | 0.19108618795871735 | 2.0687390765518066 | not_requested | 65102424576 |
| P7 | 4 | 8 | PASS | 0.19443804025650024 | 2.0687390765518066 | not_requested | 65102424576 |
| P7 | 5 | 8 | PASS | 0.1963159292936325 | 2.0687390765518066 | not_requested | 65102424576 |
| P7 | 6 | 8 | PASS | 0.18894660472869873 | 2.0687390765518066 | not_requested | 65102424576 |
| P7 | 7 | 8 | PASS | 0.19393062591552734 | 2.0687390765518066 | not_requested | 65102424576 |

Policy: bf16 DiT/LoRA, fp32 VAE and loss/reduction, safe SDPA/math policy, xformers/flash disabled, GradScaler disabled.

Warnings: expandable_segments unsupported and DDP find_unused_parameters=True warnings matched prior Exp43 behavior and did not block finite loss/gradients.

No training run, no optimizer step, no PAI write/GPU, and no quality claim occurred.
