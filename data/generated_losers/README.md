# Generated Losers

Offline generated losers should be written here or to the path in `GENERATED_LOSER_ROOT`.

Subdirectories:

- `fullmask/`: losers generated with a full mask.
- `partialmask_comp/`: partial-mask raw losers composited back onto the winner outside the mask.
- `partialmask_nocomp/`: partial-mask raw outputs used directly as losers.

Every generation run must save a manifest with at least:

`sample_id`, `prompt`, `win_video_path`, `raw_loser_video_path`, `comp_loser_video_path`, `final_loser_video_path`, `mask_path`, `mask_mode`, `mask_convention`, `comp`, `generation_model`, `source_dataset`, `seed`, `fps`, `num_frames`, `height`, `width`.
