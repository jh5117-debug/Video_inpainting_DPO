import importlib


def test_void_sft_wrapper_imports():
    mod = importlib.import_module("exp50_pai_void_adapter_feasibility.void_preference_wrapper.void_sft_wrapper")
    assert hasattr(mod, "make_target_pack")
