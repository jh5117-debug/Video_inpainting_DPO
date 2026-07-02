# Exp60B VPData Download Unblock Decision

Status: `EXP60B_VPDATA_SUBSET_BLOCKED_SOURCE_URLS`

## Answers

1. H20 hf-mirror worked for Hugging Face metadata access and downloaded
   1,089/1,100 locked Pexels videos.
2. H20 clash proxy ran, but it did not recover the remaining 11 Pexels URLs.
3. HAL fallback was needed and was run as a missing-URL probe before duplicating
   the already downloaded 1,089 videos.
4. The exact train1000/test100 subset was not completed: 1,089/1,100 videos are
   available, with 11 source URLs still blocked.
5. Full VPData was avoided.
6. Files were not transferred to PAI/NAS as a ready subset because the ready
   gate requires all 1,100 locked rows.
7. SHA256 rows exist for the 1,089 H20-downloaded files; there is no complete
   1,100-row SHA256 set.
8. PAI manifests were not generated because that would incorrectly mark an
   incomplete data root as ready.
9. Exp60B is not unblocked for PAI D3 mask generation.
10. Exact blocker: 11 locked `videos.pexels.com` raw-video URLs return HTTP 403
    from H20 hf-mirror route, H20 clash proxy route, and HAL missing-URL probe.

## Scientific / Process Boundary

No mask generation, loser generation, inference, DPO, training, GPU use, or
full VPData download was run. No row was replaced and no VPData validation claim
is made.
