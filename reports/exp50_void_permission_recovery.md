# Exp50 VOID Permission Recovery

Status: `VOID_ASSET_PERMISSION_RECOVERED`.

Checked at: `2026-06-30T02:35:12.817557+00:00`

Root-side minimal permission fix was provided by the user before this milestone. Codex did not run chmod/chown or any root permission script.

## Result

| Path | Owner | Mode | Read | Write | Execute | Write Probe |
| --- | --- | --- | --- | --- | --- | --- |
| `/mnt/nas/hj/H20_Video_inpainting_DPO/third_party/VOID` | `hj:hj` | `2770` | True | True | True | True |
| `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/void` | `hj:hj` | `2770` | True | True | True | True |
| `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/void` | `hj:hj` | `2770` | True | True | True | True |
| `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp50_pai_void_adapter_feasibility` | `hj:hj` | `2770` | True | True | True | True |
| `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp50_pai_void_adapter_feasibility` | `hj:hj` | `2770` | True | True | True | True |
| `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp50_pai_void_adapter_feasibility` | `hj:hj` | `2770` | True | True | True | True |

## Notes

- Probe files were created and deleted under each directory using `hj` identity.
- No ROSE action was performed.
- No model/code download was performed in B0.
- No training or optimizer step was run.
