# Exp26 Gate64 Readback and Mixed-Mask Protocol

Status: `GATE64_PROTOCOL_LOCKED_PENDING_PAI_GENERATION`

## Source State

- train source manifest SHA256: `68a862c18ad46271757d96c6b5483c76aeb2cfdfb5cae21a452e215b735daaa6`
- Gate64 manifest SHA256: `b904be82d58ab7cd897c6759b7351e262f61397d9f90d84df05ae42300dbffb6`
- Gate64 rows: `64`
- unique scene groups: `64`
- PAI status during this milestone: `blocked_host_key_changed_ed25519_SHA256_xDOCAS_fw0Bs5m9HizeRi1mkYOcIotlm4CxcfWwpqk`

## Distribution
### mask_profile
- `edge_touch_freeform`: 8
- `ellipse_circle_subset`: 8
- `irregular_freeform`: 16
- `object_like_polygon`: 16
- `soft_blob`: 8
- `thin_structure_freeform`: 8

### area_bucket
- `large`: 16
- `medium`: 32
- `small`: 16

### motion_bucket
- `high`: 16
- `low`: 16
- `medium`: 32

### deformation_bucket
- `moderate`: 32
- `slow`: 32

### source_kind
- `BLENDER`: 8
- `REAL`: 56

## Mask Source Audit

- Historical BR masks from YouTube-VOS K4 / Exp10-Exp11 style are not ellipse-only: area mean about 0.254, bbox height p50 about 0.791, edge-touch ratio about 0.223.
- Probe4/Gate16 masks were synthetic ellipse/circle masks; Gate16 passed, but that protocol is too narrow for Gate64.
- `vp2_mixed_br_mask_v1` therefore includes irregular free-form, object-like polygon, soft blob, edge-touch, thin-structure, and a small ellipse/circle subset.

## Locked Config

- config: `exp26_videopainter_dpo_v2/configs/vp2_mixed_br_mask_v1.json`
- manifest: `exp26_videopainter_dpo_v2/manifests/vp2_gate64_source_manifest.jsonl`

## Banned Actions

- Do not rerun Gate16 or replace the retained failed Gate16 row.
- Do not start VideoPainter DPO until Gate64 generation, full metrics, and full video review pass.
- Do not adjust this mask distribution after Gate64 generation starts.

## Next Milestone

After a fresh readback and PAI SSH restoration, run Gate64 official VideoPainter generation from this locked manifest/config.

## Config Summary

```json
{
  "area_buckets": {
    "large": [
      0.28,
      0.36
    ],
    "medium": [
      0.18,
      0.27
    ],
    "small": [
      0.08,
      0.14
    ]
  },
  "deformation_buckets": {
    "moderate": {
      "rotation_degrees": [
        12,
        35
      ],
      "scale_oscillation": [
        0.06,
        0.12
      ]
    },
    "slow": {
      "rotation_degrees": [
        0,
        12
      ],
      "scale_oscillation": [
        0.02,
        0.06
      ]
    }
  },
  "first_frame_gt": true,
  "formal_num_frames": 49,
  "hard_constraints": [
    "exactly 49 real frames per source",
    "one mask per source",
    "one seed per source",
    "scene-group disjoint sources",
    "first frame GT: frame0 mask area must be zero",
    "no VOR-Eval, no Exp26 search-dev/shadow-dev, no Exp25 search/shadow rows",
    "formal Gate64 generation must use this config without changing mask distribution"
  ],
  "historical_mask_audit": "reports/exp26_br_mask_distribution_audit_fast512.csv",
  "mask_families": {
    "edge_touch_freeform": {
      "count": 8,
      "description": "free-form regions intentionally touching one frame edge"
    },
    "ellipse_circle_subset": {
      "count": 8,
      "description": "small controlled subset retained for continuity with Gate16"
    },
    "irregular_freeform": {
      "count": 16,
      "description": "multi-lobe free-form brush-like regions"
    },
    "object_like_polygon": {
      "count": 16,
      "description": "compact non-elliptic object-like polygon masks"
    },
    "soft_blob": {
      "count": 8,
      "description": "smooth but non-circular blob masks"
    },
    "thin_structure_freeform": {
      "count": 8,
      "description": "elongated or thin free-form regions"
    }
  },
  "motion_buckets": {
    "high": {
      "centroid_motion_fraction": [
        0.18,
        0.28
      ]
    },
    "low": {
      "centroid_motion_fraction": [
        0.02,
        0.06
      ]
    },
    "medium": {
      "centroid_motion_fraction": [
        0.08,
        0.16
      ]
    }
  },
  "name": "vp2_mixed_br_mask_v1",
  "seed": 20260624,
  "source_manifest": "exp26_videopainter_dpo_v2/manifests/vp2_vor_bg_train_source_128.jsonl",
  "source_of_truth": {
    "decision": "Gate64 must be mixed-mask and cannot be ellipse/circle-only.",
    "gate16": "VideoPainter official 49F Gate16 used ellipse/circle masks and passed with one retained rejection",
    "historical_br": "YouTube-VOS D2 partial-mask K4 / Exp10-Exp11 style masks, sampled through selected_primary_comp.repaired.pai_paths.jsonl"
  },
  "status": "LOCKED_GATE64_PROTOCOL_PENDING_GENERATION"
}
```
