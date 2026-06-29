from pathlib import Path


def test_exp46_registry_exists():
    root = Path(__file__).resolve().parents[2]
    assert (root / 'experiment_registry' / 'exp46_h20_minimax_pseudosuccess_sft' / 'status.md').exists()
