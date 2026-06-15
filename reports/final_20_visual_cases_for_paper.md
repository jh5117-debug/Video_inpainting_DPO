# Final 20 Visual Cases For Paper/PPT

Selection rule: 5 verified DAVIS positive examples plus the top 15 YouTubeVOS100 candidates by metric gain, after contact-sheet sanity review.

| # | Dataset | Video | Category | Delta PSNR | Delta Mask PSNR | Delta LPIPS | Why show it |
|---:|---|---|---|---:|---:|---:|---|
| 1 | DAVIS50 | boat | boat/water boundary | +1.4362 | +1.4362 |  | strongest visual: cleaner wake/hull boundary |
| 2 | DAVIS50 | rhino | animal boundary | +0.9862 | +0.9862 |  | foreground animal mask boundary improves |
| 3 | DAVIS50 | dog-agility | animal/sports motion | +0.7001 | +0.7001 |  | large metric gain and useful motion case |
| 4 | DAVIS50 | lucia | person | +0.3976 | +0.3976 |  | subtle positive human case |
| 5 | DAVIS50 | blackswan | animal/water | +0.4102 | +0.4102 |  | mild positive water/animal boundary case |
| 6 | YouTubeVOS100 | 5b33c701ce | person/animal outdoor | +5.0176 | +5.0176 | +0.0022 | top YouTubeVOS gain candidate: dPSNR=5.018, dSSIM=0.0016, dLPIPS=0.0022 |
| 7 | YouTubeVOS100 | 8d55a5aebb | person/object indoor | +3.3585 | +3.3585 | +0.0003 | top YouTubeVOS gain candidate: dPSNR=3.358, dSSIM=0.0035, dLPIPS=0.0003 |
| 8 | YouTubeVOS100 | 990d358980 | person/animal foreground | +2.9119 | +2.9119 | -0.0015 | top YouTubeVOS gain candidate: dPSNR=2.912, dSSIM=0.0042, dLPIPS=-0.0015 |
| 9 | YouTubeVOS100 | 860c0a7cf8 | urban people / street | +2.2368 | +2.2368 | -0.0047 | top YouTubeVOS gain candidate: dPSNR=2.237, dSSIM=0.0033, dLPIPS=-0.0047 |
| 10 | YouTubeVOS100 | 1e458b1539 | person / water | +1.9564 | +1.9564 | -0.0008 | top YouTubeVOS gain candidate: dPSNR=1.956, dSSIM=0.0035, dLPIPS=-0.0008 |
| 11 | YouTubeVOS100 | c5b94822e3 | close-up texture / object | +1.8475 | +1.8475 | -0.0007 | top YouTubeVOS gain candidate: dPSNR=1.848, dSSIM=0.0086, dLPIPS=-0.0007 |
| 12 | YouTubeVOS100 | b0313efe37 | vehicle / street | +1.7815 | +1.7815 | -0.0030 | top YouTubeVOS gain candidate: dPSNR=1.781, dSSIM=0.0065, dLPIPS=-0.0030 |
| 13 | YouTubeVOS100 | 3e2336812c | large motion / landscape | +1.6819 | +1.6819 | -0.0005 | top YouTubeVOS gain candidate: dPSNR=1.682, dSSIM=0.0014, dLPIPS=-0.0005 |
| 14 | YouTubeVOS100 | eda3a7bbb1 | underwater / background texture | +1.6117 | +1.6117 | -0.0034 | top YouTubeVOS gain candidate: dPSNR=1.612, dSSIM=0.0120, dLPIPS=-0.0034 |
| 15 | YouTubeVOS100 | af881cd801 | plant / thin structure | +1.6448 | +1.6448 | -0.0010 | top YouTubeVOS gain candidate: dPSNR=1.645, dSSIM=0.0056, dLPIPS=-0.0010 |
| 16 | YouTubeVOS100 | f00dc892b2 | person / low-light indoor | +1.4670 | +1.4670 | +0.0031 | top YouTubeVOS gain candidate: dPSNR=1.467, dSSIM=0.0051, dLPIPS=0.0031 |
| 17 | YouTubeVOS100 | 966c4c022e | animal / occlusion | +1.2583 | +1.2583 | -0.0025 | top YouTubeVOS gain candidate: dPSNR=1.258, dSSIM=0.0051, dLPIPS=-0.0025 |
| 18 | YouTubeVOS100 | dcd3e1b53e | person / snow motion | +1.2923 | +1.2923 | -0.0003 | top YouTubeVOS gain candidate: dPSNR=1.292, dSSIM=0.0004, dLPIPS=-0.0003 |
| 19 | YouTubeVOS100 | e0daa3b339 | person / outdoor motion | +1.2375 | +1.2375 | -0.0011 | top YouTubeVOS gain candidate: dPSNR=1.238, dSSIM=0.0052, dLPIPS=-0.0011 |
| 20 | YouTubeVOS100 | 4c269afea9 | water / long background | +1.2457 | +1.2457 | -0.0014 | top YouTubeVOS gain candidate: dPSNR=1.246, dSSIM=0.0030, dLPIPS=-0.0014 |

## Paths

- Final local package: `/home/hj/dpo-2-1-exp/final_20_visual_cases_for_paper`
- Candidate package: `/home/hj/dpo-2-1-exp/final_20_visual_cases_for_paper_candidates`
- Each case contains a four-column side-by-side video, contact sheet, selected frames, and per-video metric row.
