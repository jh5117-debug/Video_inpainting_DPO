# Exp26 Gate64 Official Generation Status - 2026-06-25

Status: `GATE64_GENERATION_PARTIAL_SOURCE_PASS_56_OF_64`

The overnight Exp26 VideoPainter Gate64 official generation run completed on
PAI. The run did not produce 64 usable rows because the formal 49-frame guard
rejected 8 locked Gate64 sources with duplicate decoded frames. The remaining
56 rows generated successfully with the official VideoPainter stack.

## PAI Run

- PAI host: `dsw-753014-85f54df947-bkp7h`
- user: `hj`
- GPU: `0`
- parent PID / PGID: `6803`
- generation PID: `28294`
- run root: `/home/hj/exp26_gate64_runs/gate64_official_43597cf_20260625_031155`
- log: `/home/hj/exp26_gate64_runs/gate64_official_43597cf_20260625_031155.log`
- code snapshot: `/home/hj/runtime_code_snapshots/exp26_43597cf66c106ceddcdb384ec7207993662d3f1e`
- branch commit: `43597cf66c106ceddcdb384ec7207993662d3f1e`

## Pipeline Counts

| Stage | Rows | Status |
| --- | ---: | --- |
| VOR BG selective extraction | 64 | 64 OK |
| formal 49F materialization | 64 | 56 OK, 8 FAILED |
| mixed moving mask generation | 56 | 56 OK |
| official VideoPainter generation | 56 | 56 OK |

The final VideoPainter summary reports:

- `num_rows`: 56
- `ok`: 56
- `num_frames`: 49
- `num_inference_steps`: 20
- `guidance_scale`: 6.0
- status: `passed`

Outputs produced:

- raw frame directories: 56
- comp frame directories: 56
- side-by-side videos: 56
- contact sheets: 56

GPU0 was released after completion; all GPUs were at 0 MiB when checked after
the run.

## Formal 49F Rejections

The following locked Gate64 rows were rejected by the strict formal 49-frame
materializer because decoded frame images were duplicated:

- `vp2_gate64_002_REAL_ENV158_00005_001_04`
- `vp2_gate64_005_REAL_ENV233_00101_005_05`
- `vp2_gate64_013_REAL_ENV280_00103_004_05`
- `vp2_gate64_026_REAL_ENV243_00003_002_05`
- `vp2_gate64_039_REAL_ENV280_00102_007_02`
- `vp2_gate64_043_REAL_ENV185_00009_005_05`
- `vp2_gate64_048_REAL_ENV202_00005_003_04`
- `vp2_gate64_056_REAL_ENV198_00003_006_02`

These are source validity failures under the formal 49F protocol, not
VideoPainter inference failures.

## Local Copies

Small status artifacts copied into git-tracked report space:

- `reports/exp26_gate64_generation_20260625/gate64_extraction_status.csv`
- `reports/exp26_gate64_generation_20260625/gate64_materialized_49f_status.csv`
- `reports/exp26_gate64_generation_20260625/gate64_mask_status.csv`
- `reports/exp26_gate64_generation_20260625/gate64_generation_status.csv`
- `reports/exp26_gate64_generation_20260625/gate64_generation_summary.json`

Large videos and frame outputs remain on PAI and are not committed.

## Decision

This milestone is a technical generation pass for the 56 formal-valid rows. It
does not mark Gate64 data-ready:

- Gate64 visual review is still pending for all 56 generated rows.
- The 8 rejected sources must either remain excluded with an explicit source
  failure record or be replaced only through a pre-registered source-repair
  protocol.
- No VideoPainter DPO training has started.

