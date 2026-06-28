from exp40_minimax_psnr_safe_rescue.scripts import build_localdpo_v3_pool as builder


def test_profile_rotation_limits_candidates_per_source():
    assert builder.PROFILE_ROTATION
    assert max(len(group) for group in builder.PROFILE_ROTATION) <= 3


def test_known_invalid_includes_vor_quarantine():
    assert "BLENDER_RIVER007_00001" in builder.KNOWN_INVALID_SAMPLE_IDS
