import tempfile
import unittest

import numpy as np

from exp20_autoresearch_scale_adaptive_region_dpo.code.boundary_maps import DistanceCache


class DistanceCacheReloadTests(unittest.TestCase):
    def test_cached_distance_reloads_as_hit(self):
        mask = np.zeros((2, 12, 12), dtype=np.uint8)
        mask[:, 4:8, 4:8] = 1
        with tempfile.TemporaryDirectory() as tmp:
            cache = DistanceCache(tmp)
            first = cache.get(mask, identity="clip")
            self.assertEqual(cache.stats()["cache_misses"], 1.0)
            cache2 = DistanceCache(tmp)
            second = cache2.get(mask, identity="clip")
            self.assertEqual(cache2.stats()["cache_hits"], 1.0)
            self.assertTrue(np.allclose(first, second))


if __name__ == "__main__":
    unittest.main()
