import io
import tarfile
import unittest

from exp25_vor_or_preference_data.scripts.vor_archive_utils import sample_id_from_member, unsafe_member_reason


def info(name: str, *, linkname: str = "") -> tarfile.TarInfo:
    t = tarfile.TarInfo(name)
    if linkname:
        t.type = tarfile.SYMTYPE
        t.linkname = linkname
    else:
        t.size = 0
    return t


class TestVORArchiveSafety(unittest.TestCase):
    def test_reject_path_traversal(self):
        self.assertEqual(unsafe_member_reason(info("../bad")), "path_traversal")

    def test_reject_absolute_path(self):
        self.assertEqual(unsafe_member_reason(info("/abs/bad")), "absolute_path")

    def test_reject_unsafe_symlink(self):
        self.assertEqual(unsafe_member_reason(info("x/link", linkname="/etc/passwd")), "unsafe_link")

    def test_sample_id_heuristic(self):
        self.assertEqual(sample_id_from_member("VOR-Train/000123/frame_000.png"), "000123/frame_000.png")

    def test_sample_id_skips_vor_role_dirs(self):
        self.assertEqual(sample_id_from_member("VOR-Train/FG_BG/REAL_ENV900_00001_001_03.mp4"), "REAL_ENV900_00001_001_03")
        self.assertEqual(sample_id_from_member("VOR-Train/BG/REAL_ENV900_00001_001_03.mp4"), "REAL_ENV900_00001_001_03")
        self.assertEqual(sample_id_from_member("MASK/REAL_ENV900_00001_001_03.mp4"), "REAL_ENV900_00001_001_03")


if __name__ == "__main__":
    unittest.main()
