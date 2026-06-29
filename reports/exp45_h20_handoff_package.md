# Exp45 H20 Handoff Package

Status: `EXP45_H20_FILELIST_PARTIAL_SOURCE_ROOT_UNAVAILABLE`

## Scope

- PAI generated this filelist only.
- PAI did not copy files to H20.
- PAI did not validate H20 paths.
- PAI did not run training or optimizer steps.

## Source And Target

- PAI source root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp44_pai_minimax_targeted_same_source_mining/targeted_mining_20260629_161742`
- expected H20 target root: `/home/hj/H20_Video_inpainting_DPO_h20_mirror/exp44_pai_minimax_targeted_same_source_mining/targeted_mining_20260629_161742`
- required path count: `262`
- paths ready in this session: `0` files, `0` directories
- paths missing in this session: `262`
- total ready file size bytes: `0`

## Files

- filelist: `reports/exp45_h20_required_filelist.txt`
- sha256/status list: `reports/exp45_h20_required_sha256.txt`
- csv: `reports/exp45_h20_required_filelist.csv`
- json: `reports/exp45_h20_handoff_package.json`

## H20 Mirror Command Template

Run this only from a later H20 session, not from PAI:

```bash
rsync -aH --info=progress2 <PAI_HOST>:/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp44_pai_minimax_targeted_same_source_mining/targeted_mining_20260629_161742/ /home/hj/H20_Video_inpainting_DPO_h20_mirror/exp44_pai_minimax_targeted_same_source_mining/targeted_mining_20260629_161742/
```

## Current Blocker

The current session cannot see `/mnt/nas` or `/mnt/workspace`, so absolute PAI/NAS artifacts are recorded as missing here and their SHA256 cannot be computed in this session.

## Next H20 Experiment

After a separate H20 session mirrors and validates the files, the first training experiment should be pseudo-success SFT 30-step. Do not start GT-only SFT first.
