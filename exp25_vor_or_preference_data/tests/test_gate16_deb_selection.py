import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


class Gate16DebSelectionTest(unittest.TestCase):
    def test_selection_balances_sources_and_excludes_prior_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = []
            for idx in range(12):
                train.append(
                    {
                        "sample_id": f"REAL_{idx:02d}",
                        "scene_group": f"REAL_G{idx:02d}",
                        "source_type": "REAL",
                        "task": "object_removal",
                        "condition_member_path": f"VOR-Train/FG_BG/REAL_{idx:02d}.mp4",
                        "winner_member_path": f"VOR-Train/BG/REAL_{idx:02d}.mp4",
                        "mask_member_path": f"MASK/REAL_{idx:02d}.mp4",
                        "hard_comp": False,
                    }
                )
            for idx in range(12):
                train.append(
                    {
                        "sample_id": f"BLENDER_{idx:02d}",
                        "scene_group": f"BLENDER_G{idx:02d}",
                        "source_type": "BLENDER",
                        "task": "object_removal",
                        "condition_member_path": f"VOR-Train/FG_BG/BLENDER_{idx:02d}.mp4",
                        "winner_member_path": f"VOR-Train/BG/BLENDER_{idx:02d}.mp4",
                        "mask_member_path": f"MASK/BLENDER_{idx:02d}.mp4",
                        "hard_comp": False,
                    }
                )
            _write_jsonl(root / "train.jsonl", train)
            _write_jsonl(root / "root.jsonl", [{"sample_id": "REAL_00", "scene_group": "REAL_G00"}])
            _write_jsonl(root / "search.jsonl", [{"sample_id": "REAL_01", "scene_group": "REAL_G01"}])
            _write_jsonl(root / "shadow.jsonl", [{"sample_id": "BLENDER_00", "scene_group": "BLENDER_G00"}])
            _write_jsonl(root / "gate32.jsonl", [{"sample_id": "BLENDER_01", "scene_group": "BLENDER_G01"}])

            out = root / "gate16.jsonl"
            cp = subprocess.run(
                [
                    sys.executable,
                    "exp25_vor_or_preference_data/scripts/select_gate16_deb_sources.py",
                    "--train-source-pool", str(root / "train.jsonl"),
                    "--search-dev", str(root / "search.jsonl"),
                    "--shadow-dev", str(root / "shadow.jsonl"),
                    "--root-cause-manifest", str(root / "root.jsonl"),
                    "--gate32-materialized", str(root / "gate32.jsonl"),
                    "--output-manifest", str(out),
                    "--audit-json", str(root / "audit.json"),
                    "--audit-csv", str(root / "audit.csv"),
                    "--audit-md", str(root / "audit.md"),
                ],
                cwd=Path(__file__).resolve().parents[2],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            self.assertEqual(cp.returncode, 0, cp.stdout)
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows), 16)
            self.assertEqual(sum(1 for row in rows if row["source_type"] == "REAL"), 8)
            self.assertEqual(sum(1 for row in rows if row["source_type"] == "BLENDER"), 8)
            self.assertNotIn("REAL_G00", {row["scene_group"] for row in rows})
            self.assertNotIn("REAL_G01", {row["scene_group"] for row in rows})
            self.assertNotIn("BLENDER_G00", {row["scene_group"] for row in rows})
            self.assertNotIn("BLENDER_G01", {row["scene_group"] for row in rows})
            self.assertTrue(all(row["loser_stack_id"] == "DE-B_sft_raw6_d8_propainter" for row in rows))
            self.assertTrue(all(row["hard_comp"] is False for row in rows))

    def test_selection_fills_limit_when_one_source_type_is_short(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = []
            for idx in range(20):
                train.append(
                    {
                        "sample_id": f"REAL_{idx:02d}",
                        "scene_group": f"REAL_G{idx:02d}",
                        "source_type": "REAL",
                        "condition_member_path": f"VOR-Train/FG_BG/REAL_{idx:02d}.mp4",
                        "winner_member_path": f"VOR-Train/BG/REAL_{idx:02d}.mp4",
                        "mask_member_path": f"MASK/REAL_{idx:02d}.mp4",
                    }
                )
            for idx in range(4):
                train.append(
                    {
                        "sample_id": f"BLENDER_{idx:02d}",
                        "scene_group": f"BLENDER_G{idx:02d}",
                        "source_type": "BLENDER",
                        "condition_member_path": f"VOR-Train/FG_BG/BLENDER_{idx:02d}.mp4",
                        "winner_member_path": f"VOR-Train/BG/BLENDER_{idx:02d}.mp4",
                        "mask_member_path": f"MASK/BLENDER_{idx:02d}.mp4",
                    }
                )
            _write_jsonl(root / "train.jsonl", train)
            for name in ("root", "search", "shadow", "gate32"):
                _write_jsonl(root / f"{name}.jsonl", [])

            out = root / "gate16.jsonl"
            audit = root / "audit.json"
            cp = subprocess.run(
                [
                    sys.executable,
                    "exp25_vor_or_preference_data/scripts/select_gate16_deb_sources.py",
                    "--train-source-pool", str(root / "train.jsonl"),
                    "--search-dev", str(root / "search.jsonl"),
                    "--shadow-dev", str(root / "shadow.jsonl"),
                    "--root-cause-manifest", str(root / "root.jsonl"),
                    "--gate32-materialized", str(root / "gate32.jsonl"),
                    "--output-manifest", str(out),
                    "--audit-json", str(audit),
                    "--audit-csv", str(root / "audit.csv"),
                    "--audit-md", str(root / "audit.md"),
                ],
                cwd=Path(__file__).resolve().parents[2],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            self.assertEqual(cp.returncode, 0, cp.stdout)
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows), 16)
            self.assertEqual(sum(1 for row in rows if row["source_type"] == "BLENDER"), 4)
            self.assertEqual(len({row["scene_group"] for row in rows}), 16)
            audit_data = json.loads(audit.read_text(encoding="utf-8"))
            self.assertEqual(audit_data["balance_status"], "best_available_after_exclusions")


if __name__ == "__main__":
    unittest.main()
