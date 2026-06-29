# Exp45 H20 Handoff Package

Status: EXP45_H20_FILELIST_READY

## Scope

- PAI generated this filelist only.
- PAI did not copy files to H20.
- PAI did not run validation, training, DPO, SFT, or optimizer steps.

## Source And Target

- PAI source root: /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp45_pai_minimax_pair_scaleup/scaleup_mining_20260629_190229
- expected H20 target root: /home/hj/H20_Video_inpainting_DPO_h20_mirror/exp45_pai_minimax_pair_scaleup/scaleup_mining_20260629_190229
- required path count: 326
- files ready: 232
- directories ready: 94
- missing paths: 0
- total ready file size bytes: 99685806

## Formal Manifests

- manifest rows scanned: 560
- formal split: 64/24/24
- first H20 experiment: pseudo-success SFT 30-step
- do not start GT-only SFT first

## Files

- filelist: reports/exp45_h20_required_filelist.txt
- sha256/status: reports/exp45_h20_required_sha256.txt
- csv: reports/exp45_h20_required_filelist.csv
- json: reports/exp45_h20_handoff_package.json

## H20 Mirror Command Template

Run this only from a later H20 session, not from PAI:

```bash
rsync -aH --info=progress2 <PAI_HOST>:/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp45_pai_minimax_pair_scaleup/scaleup_mining_20260629_190229/ /home/hj/H20_Video_inpainting_DPO_h20_mirror/exp45_pai_minimax_pair_scaleup/scaleup_mining_20260629_190229/
```
