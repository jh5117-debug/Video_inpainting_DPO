# Exp15 DAVIS50 vs DAVIS90 Audit

## HAL / PAI Dataset Counts

- HAL source: `/home/hj/Video_inpainting_DPO/data/external/davis_2017_full_resolution/DAVIS`
  - `JPEGImages/Full-Resolution`: 90 videos
  - `Annotations/Full-Resolution`: 90 videos

- PAI full source: `/mnt/nas/hj/data/external/davis_2017_full_resolution/DAVIS`
  - `JPEGImages/Full-Resolution`: 90 videos
  - `Annotations/Full-Resolution`: 90 videos

- PAI Exp15 eval subset: `/mnt/nas/hj/data/external/davis_2017_full_resolution_or_eval50/DAVIS`
  - `JPEGImages/Full-Resolution`: 50 videos
  - `Annotations/Full-Resolution`: 50 videos

## Conclusion

MiniMax-Remover Table 2 reports all 90 DAVIS videos. Exp15 currently evaluates a DAVIS50 subset selected to match the existing project DAVIS50 protocol. Therefore Exp15 cannot be described as a MiniMax Table 2 reproduction.

## Next Step If Paper Alignment Is Required

Create a DAVIS90 manifest from the full DAVIS2017 source and rerun only after:

1. fixed visual grids are verified;
2. MiniMax-compatible metric naming is adopted;
3. paper-compatible TC and GPT-O3 VQ/Succ are either implemented or marked unavailable.

## Prepared Future Manifest

A DAVIS90 manifest was prepared but not run:

```text
exp15_or_benchmark_davis90/manifests/davis90_or_manifest.csv
```

It contains 90 videos and 6208 aligned frames using PAI full-resolution DAVIS2017 paths.
