# Exp20 Autoresearch Design

Exp20 implements a conservative, project-specific autoresearch controller.

## Hard Constraints

- Python training code is frozen after parity.
- Search changes only immutable JSON/YAML config.
- Same evaluator, seed, manifest, comp, mask, scheduler, and output protocol.
- Append-only `results.tsv`.
- Config hash de-duplicates trials.
- Crash nodes can be debugged once and then marked crash/blocked.
- DAVIS50 / YouTubeVOS100 are not used for iterative search.

## Search Space

Initial fixed image-space radii:

```text
2, 4, 6, 8, 12, 16, 24, 32, 48 px
```

Boundary contributions:

```text
0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8
```

Aggregations:

```text
legacy_global_weighted_mean
region_balanced
```

Adaptive modes:

```text
adaptive_area_perimeter
adaptive_sqrt_area
```

## Current Execution State

Root configs can be generated with:

```bash
python exp20_autoresearch_scale_adaptive_region_dpo/code/search_controller.py --init-roots
```

Heavy training remains disabled until parity and dev baseline reports pass.
