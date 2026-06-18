# Exp19c Visual Case Judgement

Visual source:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp19c_light_warp_davis10/contact_sheets/
```

Local inspection cache:

```text
/home/hj/exp19c_vis_tmp/contact_sheets/
```

Columns inspected:

```text
GT / mask / Exp11 / Exp19b / lambda000 / best lambda020
```

Judgement:

```text
better: 0
tie: 9 inspected cases
worse: 0
```

Case notes:

| Video | Motion bin | Judgement | Notes |
| --- | --- | --- | --- |
| dog-agility | high | tie | Dog and pole boundaries are visually unchanged from Exp11. |
| car-roundabout | high | tie | Moving car/background structure unchanged. |
| camel | low | tie | Body and fence region unchanged. |
| soccerball | low | tie | Ball/tree/fence region unchanged. |
| boat | low | tie | Water wake and boat boundary unchanged. |
| blackswan | medium | tie | Water/feather region unchanged. |
| rhino | low | tie | Body/tree boundary unchanged. |
| lucia | low | tie | Person/grass boundary unchanged. |
| flamingo | medium | tie | Water reflection and thin legs unchanged. |

Conclusion:

Exp19c does not introduce visible degradation, but it also does not produce
clear temporal or boundary improvements. This fails the qualitative component
of the positive gate.
