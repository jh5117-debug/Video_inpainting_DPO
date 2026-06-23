# Exp25 VOR OR Preference Data

- repo: FudanCVL/EffectErase
- HF authenticated user: JiaHuang01
- dataset revision: `fa09dc61128ca0418a4a13364d97a08018ea9cc7`
- PAI outbound HF network: unavailable; use HAL-only download then rsync to PAI.
- core scope: README, VOR-Eval parts, VOR-Train-MASK parts, VOR-Train parts.
- excluded this round: VOR-Wild.
- required files: 37
- required total bytes: 363730944386
- largest part bytes: 10737418240
- HAL staging: `/home/hj/exp25_effecterase_staging`
- HAL free bytes at selection: 571536965632
- PAI destination: `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/downloads/fa09dc61128ca0418a4a13364d97a08018ea9cc7`
- status: CORE_DOWNLOAD_COMPLETE
- completed time: 2026-06-23T00:08:45+0200
- completed files: 37 / 37
- completed bytes: 363730944386 / 363730944386
- PAI final files: 37
- PAI partial files: 0
- PAI bad files: 0
- HAL staging final size: 1.0K

## Safety

This track is download-only. It does not enter Exp23 worktrees, use GPUs, run inference, generate losers, or start DPO training. Tokens remain only under `/home/hj/.cache/huggingface_effecterase_auth` and are not copied to PAI or committed.

## Completion Notes

The core EffectErase VOR compressed archive scope completed via HAL-only Hugging Face download and HAL-to-PAI rsync. Each file was processed serially, verified with HAL and PAI SHA256 equality before atomic finalization, and the per-file HAL job/cache was removed after verification.

The completion marker exists on PAI:

`/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/runtime/CORE_DOWNLOAD_COMPLETE`

Final PAI-side inventory verification completed successfully after transfer. The independent verifier checked all 37 required files under the fixed revision directory, reported `ok=true`, found 0 partial files and 0 bad files, and confirmed contiguous archive parts for VOR-Eval, VOR-Train-MASK, and VOR-Train. The append-only transfer manifest also contains 37 VERIFIED rows and zero HAL/PAI SHA256 mismatches from transfer-time checks.

## Next Phase: Selective OR Data Construction

The next phase must not materialize or generate losers for the full 60K VOR
training set. Exp25 now adds isolated tooling for:

- lightweight split-archive continuity and byte-count inspection;
- resumable tar member indexing with path-safety checks;
- selective extraction of VOR-Eval and chosen VOR-Train/VOR-Train-MASK sample
  IDs;
- validation of extracted subsets;
- canonical OR manifest semantics where `condition = V_obj`, `winner = V_bg`,
  `mask = foreground object mask`, `hard_comp = false`, and losers are raw
  generator outputs.

The first formal source pool remains capped at 4096 train candidate triplets,
256 search-dev triplets, and 256 shadow-dev triplets. Preference manifests must
be nested at 512/1024/2048/3072, with 4096 allowed only if 3072 remains clearly
better than 2048.
