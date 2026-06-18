# Status

```text
BLOCKED_AT_ARCHITECTURE_PREFLIGHT
```

Exp19 is isolated from old experiments. The architecture preflight found that
the shared `UNetMotionModel` residual interface is unsafe for the requested
multi-scale down+mid flow adapter. Training was not launched.
