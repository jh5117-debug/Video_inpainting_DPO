# Exp27 LocalDPO Official Mask Root Cause

Date: 2026-06-23

## Official Code Identity

Repo:

`/home/hj/video_dpo_paper_code_cache/repos/Local-DPO`

Commit:

`7528e966b17283cfa638577827e456737335f030`

File:

`innerT2V/utils/random_mask_gen.py`

## Failure

Official default call:

```python
create_random_shape_with_random_motion(
    video_length=13,
    zoomin=0.9,
    zoomout=1.1,
    rotmin=1,
    rotmax=10,
)
```

Current error:

`ValueError: cannot reshape array of size 1228800 into shape (480,640,3)`

This occurs with the official default `imageHeight=240`, `imageWidth=432`; it is not caused by our smaller Exp27 test size.

## Root Cause

Relevant official lines:

- line 214: `data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)`
- line 216: `data = data.reshape((fig.canvas.get_width_height()[::-1] + (3,)))`

In the current dependency environment:

- matplotlib version: `3.10.7`
- canvas size: `[640, 480]`
- `tostring_argb()` size: `1,228,800`
- expected RGB size: `921,600`
- expected ARGB size: `1,228,800`
- `buffer_rgba()` shape: `[480, 640, 4]`
- `tostring_rgb` is not available.

The official code reads a 4-channel ARGB buffer and reshapes it as 3-channel RGB. After fixing that conversion, a second official-file issue appears: `random_mask_gen.py` calls `cv2.resize(...)` without importing `cv2` in that file.

Classification:

`B_DEPENDENCY_MATPLOTLIB_CANVAS_MODE_DIFFERENCE`

with an additional local missing-import dependency in the official file.

## Compatibility Patch

The official clone is not modified.

Exp27 adds:

`exp27_paper_grounded_preference_study/code/localdpo_compat.py`

Patch scope:

1. Load the official `random_mask_gen.py` module.
2. Replace only `get_random_shape(...)`.
3. Convert official `tostring_argb()` output to RGB by reshaping as ARGB and dropping alpha.
4. Provide `cv2` from the wrapper for the resize call.
5. Preserve the official random shape, movement, zoom, rotation, and connected-component logic.

Compat parity output:

`exp27_paper_grounded_preference_study/parity/localdpo_compat_parity.json`

Default probe result:

- status: `passed_with_official_code_compatibility_patch`
- shape: `[13, 240, 432]`
- sha256: `7534f699da624961be58988140767e2e524ed556d639e7103e47a7605035bfa9`

Small deterministic probe:

- shape: `[4, 64, 96]`
- sha256: `583f42ea17bfda077e0b8380fa35c5fab1767162de8774653f2b4cb59a77082e`

## Current Gate Status

LocalDPO mask generation is now:

`OFFICIAL_CODE_COMPATIBILITY_PATCH_MASK_ONLY_PASSED`

This is not yet a faithful LocalDPO baseline. Remaining gates:

1. single-video corruption parity;
2. 6-video pair-generation smoke;
3. task mask vs LocalDPO corruption mask separation;
4. restoration-critical affected-region separation;
5. 1/10-step training smoke.
