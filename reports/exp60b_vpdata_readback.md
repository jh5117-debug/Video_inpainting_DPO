# Exp60B VPData Readback

Status: `EXP60B_READBACK_DONE`

Branch: `research/exp60b-videopainter-vpdata-d3mask-pai-20260702`

Start HEAD: `568a7dfb48bcdfce893176a1dd48c653414a13a8`

## Prior VideoPainter State

The previous VideoPainter evidence came from VOR-BG clean videos, Gate64
official VideoPainter self-losers, and a balanced primary32 preference set.
Reports show:

- Gate64 formal valid: 64/64.
- Eligible: 55.
- Primary32: 16 medium-hard + 16 hard-plausible.
- Step50 passed in the locked search-dev micro setting.
- Exp31 Step2000 was reported positive under the fixed VOR-BG comp protocol.

Recent visual audit also showed that old comp evidence is not enough for a
paper-facing raw-capability claim because comp pastes winner pixels outside the
mask. Exp60B therefore starts a new VPData-subset path instead of continuing
the old VOR-BG primary32 setting.

## VPData Source Readback

Official VideoPainter docs and the Hugging Face dataset page indicate:

- Dataset: `TencentARC/VPData`.
- Access: public Hugging Face dataset page was reachable from HAL.
- Dataset size shown by Hugging Face: 392,077 rows, approximately 1.87 TB.
- Modalities: csv, text, video.
- The official README says VPData/VPBench contain 390K+ mask sequences and
  video captions.
- The official README says VPData directly provides masks/text annotations;
  VideoVo raw videos are uploaded to VPData, while Pexels raw videos are
  downloaded from URLs in `pexels.csv` via `data_utils/VPData_download.py`.

## Download Boundary

Do not run a full dataset clone or the unmodified download script.

The official `data_utils/VPData_download.py` reads all rows in `pexels.csv` and
downloads each Pexels video. That is not compatible with Exp60B. The next
download step must first materialize a deterministic train1000/test100 source
list, then selectively download only required zip files and row-level Pexels
videos.

## D3 Mask Readback

The DiffuEraser main data audit confirms the D3 partial-mask setup:

- 3,327 primary-comp pairs.
- DiffuEraser-only self-rollout.
- K=4 masks per source.
- Mask convention: PNG value 255 is inpaint region; 0 is keep region.
- Canonical D3 v1 policy: 16 frames, 512x320, mask area 20%-30%, interior
  constrained irregular polygon, two static and two slow masks when K=4.

Exp60B must recover this logic exactly before creating VPData masks.

## Machine Readback

Current session host: HAL.

PAI host `hj@47.103.26.60` is reachable. During the probe, PAI GPU0 and GPU1
reported 0 MiB used and 0% utilization. GPU2-GPU7 had unrelated lightweight
processes and are out of scope.

H20 was not reachable via aliases `h20` or `pai`; `pai` was recovered through
the known IP, but no equivalent H20 route was found locally. H20 download is
therefore blocked from this session until the real H20 host/route is supplied.

## Answers Required By Milestone A

1. Is VPData publicly downloadable?
   Yes, the Hugging Face dataset page is public and reachable.

2. Does VPData require gated access?
   No gated access was observed from the public dataset page.

3. Can we selectively download train1000 + test100?
   Likely yes, but only with a custom selective downloader. A full clone or the
   unmodified official Pexels downloader would exceed the boundary.

4. If full archive is required, can we range-select or partial-sync?
   The Hugging Face tree exposes grouped mask/raw-video zip files and CSV files.
   A downloader must select only zip groups and Pexels rows needed by the locked
   subset.

5. What is official train/test split?
   Official files include `pexels_videovo_train_dataset.csv`,
   `pexels_videovo_val_dataset.csv`, and `pexels_videovo_test_dataset.csv`.
   Exp60B must sample train1000 from train and test100 from val/test without
   overlap.

6. What metadata fields exist?
   The Hugging Face viewer shows video file names, frame/time indices, mask id,
   and captions. Exact CSV columns must be captured during selective metadata
   download before sampling.

7. What captions exist?
   VPData includes dense English video captions in the metadata CSV rows.

8. What native segmentation masks exist?
   VPData includes high-quality segmentation mask sequences, stored in grouped
   Pexels/VideoVo mask zip files. These are audit-only for Exp60B.

9. Estimated storage for train1000 + test100.
   Not yet final because video length/resolution varies and Pexels raw videos
   are URL-backed. The full dataset is about 1.87 TB, so a 1100-row subset plus
   selected mask zips should be far smaller but still requires H20 free-space
   checking before download.

10. Why use DiffuEraser D3 masks, not VPData native masks?
    The goal is protocol equivalence with the main LoVI-DPO DiffuEraser line:
    shape, size, motion, K=4, mask polarity, and region semantics should match
    the existing D3 partial-mask data.

11. Why move loser generation/training to PAI GPU0/GPU1?
    PAI has large NAS-backed storage and GPU0/GPU1 were free in the readback
    probe. H20 is reserved for subset download/transfer only.

12. Why no 2000-step yet?
    This is a new data source and mask protocol. It must pass subset download,
    D3 mask equivalence, loser quality gates, larger pair selection, and 50-step
    smoke before any longer training.

