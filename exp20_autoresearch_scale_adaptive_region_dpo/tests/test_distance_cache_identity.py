import tempfile
import unittest

import numpy as np

from exp20_autoresearch_scale_adaptive_region_dpo.code.boundary_maps import DistanceCache


class DistanceCacheIdentityTests(unittest.TestCase):
    def test_key_changes_with_identity_and_mask(self):
        mask = np.zeros((2, 8, 8), dtype=np.uint8)
        mask[:, 3:5, 3:5] = 1
        changed = mask.copy()
        changed[:, 2:5, 3:5] = 1
        with tempfile.TemporaryDirectory() as tmp:
            cache = DistanceCache(tmp)
            key_a = cache.key(mask, identity="video_a:0-1")
            key_b = cache.key(mask, identity="video_b:0-1")
            key_c = cache.key(changed, identity="video_a:0-1")
        self.assertNotEqual(key_a, key_b)
        self.assertNotEqual(key_a, key_c)


if __name__ == "__main__":
    unittest.main()
