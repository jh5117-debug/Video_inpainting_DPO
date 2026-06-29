# Exp45 Qualitative Summary

Exp45 starts from the Exp44 qualitative finding that MiniMax has real
same-source success/failure signal, but the handoff remains too small for
training-unlocked Stage2 claims.

The previous H20-side path validation attempt is treated as a scope deviation.
This lane will only produce PAI-side data, filelists, checksums when possible,
and H20 mirror instructions. No H20 execution, training, optimizer step, or
MiniMax-positive claim is allowed.

Milestone B produced the H20 mirror filelist without executing any H20 action.
Because this session cannot see `/mnt/nas` or `/mnt/workspace`, the raw
MiniMax outputs, review sheets, frame directories, source frames, masks, and
winner frames are recorded as missing in the current session. This is a
handoff packaging blocker only; it does not invalidate the repo-side Exp44
manifests, whose checksums were recomputed locally.

Milestone C was rerun on the true PAI host `dsw-753014-85f54df947-bkp7h` after
the HAL environment correction. The required NAS roots were visible there, and
official MiniMax inference produced 72 new targeted candidates across six
locked source groups. Automatic scoring found 38 success candidates and 26
medium-hard failure candidates. This is still a pre-relabel mining result:
Milestone D must inspect the generated visual pages before any rows become
training-eligible.

Milestone D opened all eight PAI-generated review pages and applied a
conservative relabel policy. The accepted pool keeps 28 usable successes and
22 medium-hard failures, while rejecting too-close, fogging, and borderline
rows. This improves label purity versus the automatic mining output, but the
new-row same-source pair capacity is still lower than the raw auto count, so
formal split construction must combine the new rows with the Exp44 accepted
pool and enforce disjoint scene groups.

Milestone E built the formal Exp45 handoff by combining the Exp44 curated pool
with the new Exp45 PAI relabel output. The handoff now reaches the preferred
`64/24/24` split for pseudo-success distillation, GT distillation, and
same-source preference rows, with zero scene overlap and all referenced paths
visible on PAI. H20 is still untouched by this lane; the next H20 action is to
mirror the package and run pseudo-success SFT 30-step first, not GT-only SFT.

Paper positioning: MiniMax now has a curated same-source Stage2 handoff ready
for H20 training, but it is still not quality-positive and not third-adapter
evidence until H20 training plus shadow evaluation passes. DiffuEraser and
VideoPainter remain the only current positive adapter evidence.
