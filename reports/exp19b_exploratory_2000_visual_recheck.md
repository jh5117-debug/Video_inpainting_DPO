# Exp19b Exploratory 2000 Visual Recheck

Date: 2026-06-18

Purpose: answer whether the Exp19b exploratory 2000 DAVIS50 result has visible
quality or temporal advantages over Exp11 outer b0.75 S2.

Visual sources:

```text
/home/hj/exp19b_exploratory_davis50_contact_sheets/
/home/hj/exp19b_exploratory_davis50_contact_sheets_selected/
```

The selected folder contains cases chosen from per-video metrics:

- best Ewarp deltas: `motocross-jump`, `scooter-gray`, `paragliding`, `rhino`
- worst Ewarp regressions: `surf`, `horsejump-high`, `drift-straight`
- best PSNR deltas: `kite-walk`, `car-shadow`, `libby`
- worst PSNR regressions: `bus`, `hockey`, `lucia`, `rollerblade`

Actually reviewed in this pass:

```text
dog-agility
car-roundabout
boat
blackswan
dance-jump
train
motocross-jump
scooter-gray
kite-walk
surf
horsejump-high
bus
lucia
rollerblade
rhino
mallard-water
```

## Case Notes

| Video | Visual judgement | Notes |
| --- | --- | --- |
| dog-agility | tie / slight Exp11 preference | SFT to Exp11 improvement is visible around fence/grass and pole seams. Exp19b and Exp11 are nearly identical; no clearer dog boundary or reduced drag. |
| car-roundabout | tie | Car body, road, and building background look essentially the same. No visible motion-boundary gain. |
| boat | tie | Exp11 already fixes most water/wake haze. Exp19b does not sharpen wake, boat edge, or shoreline. |
| blackswan | tie / slight Exp11 preference | Water and feather details are close. Exp19b does not make ripples more coherent. |
| dance-jump | tie / slight Exp11 preference | Thin structures and skirt motion do not improve. Differences look like sampling/compression noise. |
| train | tie | Rails and toy train edges stay stable; Exp19b does not add visible temporal benefit. |
| motocross-jump | tie | This is the best Ewarp-improvement case numerically, but wheel, rider, and forest background look visually tied with Exp11. |
| scooter-gray | tie | Small positive metric deltas, but scooter wheels, road sign, and columns are visually unchanged. |
| kite-walk | tie | Largest PSNR gain, but mask area is small sand; no visible improvement in sand, surf, or kite-line structure. |
| surf | tie / slight Exp11 preference | Worst Ewarp regression case. No catastrophic ghosting, but no improvement around sail edge or sea texture. |
| horsejump-high | tie | Horse legs, obstacle, and dust/boundary are not better than Exp11. |
| bus | tie | Worst PSNR-regression family; bus windows and roof edge remain visually tied. |
| lucia | tie | Grass texture and person boundary are essentially unchanged. |
| rollerblade | tie | Graffiti alignment and skater edge are not improved by Exp19b. |
| rhino | tie | Tree/rocks/rhino boundary look unchanged. |
| mallard-water | tie | Water ripple continuity and duck boundary do not improve. |

## Conclusion

Visual evidence supports the quantitative conclusion:

```text
Exp19b exploratory 2000 is safe but effectively no-op relative to Exp11.
```

I did not find a convincing better-than-Exp11 case. The best metric-improvement
cases are still visually tied. The worst metric-regression cases are also not
visibly broken, which means the adapter is not harmful, but it is too weak to be
useful under this setup.

Decision:

```text
Do not continue Exp19b under the current adapter-only / Exp11-DPO-only setup.
Current best remains Exp11 outer b0.75 S2.
```
