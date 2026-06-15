# PRD 27: VideoPainter Adapter Feasibility And Smoke

Date: 2026-06-15

## Summary

VideoPainter is a feasible future adapter target, but smoke has not run.

## Smoke Status

| Step | Status | Reason |
|---|---|---|
| Repo audit | pass | local official repo found |
| Training entry audit | pass | official train scripts exist |
| Loss interface audit | design-feasible | diffusion tensors exist, DPO not implemented |
| Reference model audit | feasible-in-principle | second frozen copy possible, memory not measured |
| Smoke1 | not run | HAL session, adapter trainer missing |
| Smoke20 | not run | Smoke1 not passed |
| Gate2000 | not ready | requires smoke pass + user confirmation |

## PAI Manual Command

This command is safe; it should currently stop with a clear `BLOCKED` message
until adapter code is implemented.

```bash
cd /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO || exit 1
git pull --ff-only
bash exp14_adapter_videopainter/scripts/run_videopainter_adapter_smoke1_pai.sh
```

The guard script checks or clones:

```text
$PROJECT_ROOT/third_party/VideoPainter
```

## Next Action

Implement an isolated adapter trainer under:

```text
exp14_adapter_videopainter/code/
```

Then run Smoke1 on PAI.
