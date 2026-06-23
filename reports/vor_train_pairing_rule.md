# VOR Train Pairing Rule

Pairing is by exact video basename across `VOR-Train/FG_BG`, `VOR-Train/BG`, and `MASK`.

- complete_triplets: 57751
- incomplete_video_ids: 2473
- scene_groups: 1449
- condition = FG_BG / V_obj
- winner = BG / V_bg
- mask = MASK / foreground object mask
