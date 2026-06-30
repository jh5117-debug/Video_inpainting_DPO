# Exp50 VOID Preference Forward

Status: `VOID_PREFERENCE_FORWARD_PASS`

## Setup

- Sample: `BLENDER_CON001_00636`
- Policy/reference: identical VOID pass1 transformer clones.
- Reference frozen: True
- Trainable policy filter: `proj_out`
- Trainable parameters: 393344 / 5571462784
- Target parameterization: `v_prediction`
- Same noise/timestep: True / True

## Losses

- winner policy/reference: 0.0640571117401123 / 0.0640571117401123
- loser policy/reference: 0.08385218679904938 / 0.08385218679904938
- winner gap: 0.0
- loser gap: 0.0
- preference margin: 0.0
- DPO loss: 0.6931471824645996
- policy grad finite: True grad_norm=0.011950799647202132

## Safety

No optimizer step, training loop, VOR-Eval, hard comp, deepspeed install, or VOID positive claim was performed.
