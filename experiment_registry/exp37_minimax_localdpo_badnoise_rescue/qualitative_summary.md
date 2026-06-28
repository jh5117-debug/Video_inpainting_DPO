# Exp37 Qualitative Summary

No Exp37 videos have been generated yet.

Readback imported prior MiniMax qualitative status:

- Exp30 visual better rows: `0/32`.
- Exp35 visual better rows: `0/48`.
- Exp36 visual better rows: `0/24`.

The next qualitative question is whether train-side videos improve while
heldout does not, or whether both train and heldout remain non-positive.

## 2026-06-28 Train-vs-Heldout Diagnosis

Codex reviewed `32/32` Step0-vs-Step10 temporal strips:

- Train16: `0/16` better, `16/16` tie/no visible change, `0` new artifacts.
- Heldout16: `0/16` better, `16/16` tie/no visible change, `0` new artifacts.

Representative checked rows included the best and worst metric rows:
`BLENDER_GRASS001_00001`, `BLENDER_MOUNTAIN008_00001`,
`REAL_ENV087_00002_001_01`, and `REAL_ENV105_00004_001_01`. Step10 was
visually indistinguishable from Step0 in all checked strips; the diff columns
were near black.

Qualitative diagnosis: `MINIMAX_OBJECTIVE_SIGNAL_TOO_WEAK`.

## 2026-06-28 LocalDPO-style OR Corruption Pool

Codex opened all `48/48` selected primary review sheets in six batches. The
pool contains localized object/affected/boundary defects with far-outside
background preserved. Final visual counts:

- Medium-hard: `38`.
- Hard-but-plausible: `10`.
- Too-close: `0`.
- Trivial-bad: `0` after Codex final review.
- Technical-invalid: `0`.

No black/purple collapse, global frame damage, or systematic far-outside
breakage was observed. Several REAL human/animal rows remain intentionally
hard and are labeled `HARD_BUT_PLAUSIBLE`, not medium-hard.

## 2026-06-28 Bad-Noise Diagnostic Scan

No videos were generated in this milestone. It was a frozen-model diagnostic
forward scan over latent/noise/timestep states. The qualitative implication is
that the next recipe must avoid using arbitrary high-residual states: the
random baseline often had larger local and gradient proxies because it also
introduced outside damage. The selected `hard_state_A/B/C` states preserve the
LocalDPO goal of local difficulty with outside sanity.
