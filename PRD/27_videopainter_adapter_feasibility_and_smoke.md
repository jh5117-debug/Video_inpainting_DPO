# PRD 27: VideoPainter Adapter Feasibility And Gate Status

Date: 2026-06-15

## Summary

VideoPainter is a feasible future adapter target at the repository / model
architecture level. The user later requested skipping smoke and going directly
to a 2000-step gate. The gate precheck was attempted, but it is blocked because
the isolated adapter trainer has not been implemented.

## Smoke Status

| Step | Status | Reason |
|---|---|---|
| Repo audit | pass | local official repo found |
| Training entry audit | pass | official train scripts exist |
| Loss interface audit | design-feasible | diffusion tensors exist, DPO not implemented |
| Reference model audit | feasible-in-principle | second frozen copy possible, memory not measured |
| Smoke1 | skipped | user requested no smoke |
| Smoke20 | skipped | user requested no smoke |
| Gate2000 | blocked | isolated adapter trainer missing |

## Gate2000 Command

The gate script no longer requires smoke outputs or extra confirmation. It
still performs hard prechecks and blocks if the adapter trainer or reference /
loss interface is unavailable.

```bash
cd /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO || exit 1
git pull --ff-only
bash exp14_adapter_videopainter/scripts/launch_videopainter_adapter_gate2000_pai.sh
```

Current expected result:

```text
BLOCKED: isolated VideoPainter DPO adapter trainer is not implemented
```

## Next Action

Implement an isolated adapter trainer under:

```text
exp14_adapter_videopainter/code/
```

Then rerun gate2000 precheck. Smoke remains skipped by user request, but the
trainer and policy/reference DPO loss interface are non-negotiable.

See:

```text
PRD/28_videopainter_adapter_gate2000_result.md
reports/videopainter_adapter_gate2000_precheck.md
```
