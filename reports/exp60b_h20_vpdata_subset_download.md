# Exp60B H20 VPData Subset Download

Status: `EXP60B_H20_VPDATA_SUBSET_BLOCKED_NETWORK`

No VPData raw video was downloaded.

## What Was Attempted

The Exp60B selective downloader was pushed to the branch and pulled into the
H20 worktree:

- H20 worktree:
  `/home/nvme01/H20_Video_inpainting_DPO_exp60b_vp_vpdata_transfer`
- H20 HEAD: `ea9c3e6b51fc4ee5f675d4f1a1a26f65495151e3`
- H20 route: HAL -> PAI -> H20
- H20 storage: `/home/nvme01` has 1.2T free and passes the hard stop.

The downloader's Python module and unit tests passed on H20. The shell wrapper
was fixed to use `python3`, because `python` is not present on H20.

## Blocker

Running even a small `train10/test5` plan on H20 failed before any raw-video
download:

```text
urllib.error.URLError: <urlopen error [Errno 101] Network is unreachable>
```

The failing URL was the official VPData metadata CSV on Hugging Face. A direct
H20 probe also returned `Network is unreachable` for `https://huggingface.co`.

PAI was also checked:

- PAI -> Hugging Face: `Network is unreachable`
- PAI -> Pexels HEAD probe: HTTP 403

HAL can read the Hugging Face metadata and generated the deterministic
Pexels-only train1000/test100 plan, but Exp60B specifically scoped raw subset
download to H20. Therefore the official H20 subset download is blocked until
network egress is available or the download route is reauthorized.

## Outputs Produced

Plan-only files, no raw videos:

- `manifests/exp60b_vpdata_train1000_sources_h20.jsonl`
- `manifests/exp60b_vpdata_test100_sources_h20.jsonl`
- `reports/exp60b_vpdata_subset_plan.md`
- `reports/exp60b_vpdata_subset_plan.csv`
- `reports/exp60b_vpdata_subset_plan_summary.json`

## Next Safe Options

1. Enable H20 outbound HTTPS to Hugging Face and Pexels, then rerun the guarded
   downloader with `--download`.
2. Explicitly authorize a HAL-first or other egress-enabled download route,
   followed by checksum transfer to PAI/NAS.
3. Provide an internal mirror of the selected VPData raw videos.

Full VPData clone/download remains forbidden.

