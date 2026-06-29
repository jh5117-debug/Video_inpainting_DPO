# Exp47 H20 MiniMax Pseudo-Success Forensic

Current status: `EXP47_MANIFEST_ALIGNMENT_PASS`.

Exp47 is a forensic-only audit of the Exp46 pseudo-success SFT30 regression. No training, optimizer step, GT-only SFT, DPO, or 100-step run is allowed.

Milestone B audited `112` Exp46 runner manifest rows and found `0` failed rows. Active pseudo-success targets are H20-local `winner_path` frame directories; all active paths exist, no PAI/HAL active paths remain, frame counts/resolutions match, masks are non-empty with expected polarity, target identity is pseudo-success, VOR-Eval and hard-comp are excluded, and train/search/shadow overlap is zero.

Next audits: teacher quality, Step30 movement direction, and region loss contribution.


Milestone C teacher quality: `EXP47_TEACHER_GLOBAL_DRIFT_CONFIRMED`. Search/shadow pseudo-success targets have strict-clean `0`, local-only `22`, and global-drift `26` rows. Codex inspected all four contact pages covering 48 rows. Teacher quality is not clean enough for full-video global SFT without localization or stricter relabel.
