# Paper Evidence Status After Exp31 and Exp33

Paper claim status: `TWO_BACKBONE_PLUS_MINIMAX_PLUMBING_ONLY`

VideoPainter 2000-step strengthens the paper as qualified long-run evidence, but the formal positive gate remains blocked by missing LPIPS/Ewarp in this fast summary. EffectErase is technically evaluated on 43/43 held-out VOR-Eval rows, but quality is weak/mixed and it is not adapter evidence. MiniMax remains not a third successful adapter unless a right-plugin Exp36 result later proves otherwise.

| model | task | dataset | metric status | visual status | final role | claim allowed | claim forbidden | next experiment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DiffuEraser main BR | VOR-BG adapter | BR / DAVIS50 lineage | main positive from prior registry | reviewed in prior reports | primary method evidence | main LoVI/BR evidence | final SOTA; universal adapter | none before paper audit |
| DiffuEraser VOR-OR | object removal preference data | VOR-OR | blocked | not new | not paper-positive yet | blocked future work only | VOR-OR success claim | explicit user authorization for data gate |
| VideoPainter 50-step | cross-backbone adapter micro gate | VOR-BG search/shadow | positive on search/shadow from Exp26 | reviewed | micro cross-backbone evidence | second-backbone micro evidence | external generalization; final SOTA | none unless external follow-up authorized |
| VideoPainter 2000-step | cross-backbone adapter long-run | fixed search-dev + shadow-dev | strong available metric gains; formal positive blocked by LPIPS/Ewarp gap | all-32 search/shadow pages reviewed | qualified long-run evidence | stronger VideoPainter long-run evidence with caveat | universal adapter; final SOTA | optional LPIPS/Ewarp completion only |
| MiniMax Exp35/Exp36 | third adapter rescue | MiniMax rescue lanes | MINIMAX_RESCUE_RECIPE_NOT_READY from prior readback | not touched by left CLI | not third successful adapter | plumbing/rescue status only | third successful adapter | right plugin/user controlled |
| EffectErase VOR-Eval | held-out object-removal baseline | VOR-Eval 43 rows | weak/mixed baseline | 43/43 review sheets and 43/43 crop sheets opened through contact sheets | held-out weak baseline only | baseline diagnostic | adapter evidence; DPO loser source; strong baseline | none without separate authorization |
| Exp27 LocalDPO / SDPO / Linear-DPO | objective study | paper-grounded objective branch | not updated here | not updated here | not new paper evidence from this prompt | pending objective baseline status only | RC-FPO or O0-O5 completion claim | Exp34/Exp27 objective run after authorization |

## Direct Answers

1. VideoPainter 2000-step strengthens the paper, but as qualified long-run evidence rather than formal final positive.
2. EffectErase baseline is technically complete but weak/mixed, so it is not a strong baseline and not adapter evidence.
3. MiniMax is not a third successful adapter from this left-CLI work; it remains right-plugin territory and was not touched.
4. Universal-adapter, all-models-supported, final-SOTA, and top-conference novelty-confirmed claims remain forbidden.
5. Three-backbone successful-adapter evidence is not established. The current defensible paper framing is DiffuEraser plus VideoPainter, with MiniMax/EffectErase as incomplete or weak auxiliary baselines.
