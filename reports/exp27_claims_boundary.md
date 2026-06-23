# Exp27 Claims Boundary

## Claims We Cannot Make

- First localized DPO for video diffusion.
- First region-aware or mask-guided video DPO.
- First real-video winner plus model-generated local loser preference setup.
- First winner-preserving DPO for diffusion.
- First fix for sigmoid DPO saturation.
- First unified diffusion/flow-matching DPO.
- SDPO guarantee for LoVI without a new proof.
- Linear-DPO novelty if we use its utility or EMA reference directly.

## Claims That Remain Plausible

- Video inpainting/object removal preference optimization has task-native failure structure absent from generic text-to-video LocalDPO.
- BR and OR require different restoration-critical region definitions.
- Inpainting preference pairs need anti-shortcut checks for outside-context damage, copy-condition, and copy-winner behavior.
- Defect-balanced medium-hard loser selection may outperform random local corruption for restoration.
- Stage1 and Stage2 may need different preference data because spatial and temporal failures differ.

## Novelty Bar

The primary contribution must show at least one of:

- task-native failure-structured data beats faithful LocalDPO-style corruption;
- restoration-critical BR/OR region decomposition beats mask-only LocalDPO RA-DPO;
- stage-aware spatial/temporal preference signals beat reusing one loss/data recipe for both stages.

SDPO and Linear-DPO are required baselines/ablations, not the main claim.
