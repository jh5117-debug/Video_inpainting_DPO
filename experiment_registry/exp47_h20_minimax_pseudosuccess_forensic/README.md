# Exp47 H20 MiniMax Pseudo-Success Forensic

Current status: `EXP47_MANIFEST_ALIGNMENT_PASS`.

Exp47 is a forensic-only audit of the Exp46 pseudo-success SFT30 regression. No training, optimizer step, GT-only SFT, DPO, or 100-step run is allowed.

Milestone B audited `112` Exp46 runner manifest rows and found `0` failed rows. Active pseudo-success targets are H20-local `winner_path` frame directories; all active paths exist, no PAI/HAL active paths remain, frame counts/resolutions match, masks are non-empty with expected polarity, target identity is pseudo-success, VOR-Eval and hard-comp are excluded, and train/search/shadow overlap is zero.

Next audits: teacher quality, Step30 movement direction, and region loss contribution.
