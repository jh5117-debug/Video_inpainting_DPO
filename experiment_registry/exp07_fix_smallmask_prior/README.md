# Exp7 fix: VideoDPO small-mask 15-20 + ProPainter prior

- status: `h20_s1s2_launching`
- short_name: `fix_smallmask_prior`
- task: H20 Stage1+Stage2 partial-mask inpainting run
- data: new non-overwriting VideoDPO small-mask generated losers

## Loss

m_w = policy winner MSE; m_l = policy loser MSE; m_w_ref/m_l_ref are reference MSE.
win_gap = m_w - m_w_ref; lose_gap = m_l - m_l_ref.
inside = -0.5 * beta_dpo * (win_gap - lose_gap_weight * lose_gap).
L_DPO = mean[-logsigmoid(inside)].

Experiment-specific loss: L = -logσ{-0.5 * 10 * (win_gap - 0.25 * lose_gap)} + 0.05 * m_w + ReLU(win_gap). Prior/data fix: small mask 15%-20% and ProPainter prior.

## Conclusion

Planned correction for Exp7 before expanding D3.

## Next Action

H20 smallmask data is available. Launch Stage1+Stage2 with
`experiment_registry/exp07_fix_smallmask_prior/code/launch_s1s2_h20.sh` from a
clean git-synced H20 worktree, using GPUs 1-7 and the fp32 SIGFPE-safe profile.

## Code Ownership

Canonical experiment-specific code lives under this registry folder:

- `code/launch_s1s2_h20.sh`
- `code/posthoc_davis_val_h20.sh`
- `code/smallD2_two_variant_eval_sharded_h20.sh`

The matching top-level `scripts/` entries are compatibility wrappers only.
