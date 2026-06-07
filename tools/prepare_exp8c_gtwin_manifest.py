#!/usr/bin/env python3
"""Compatibility wrapper for the Exp8c registry-local manifest tool."""

from pathlib import Path
import runpy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = (
    PROJECT_ROOT
    / "experiment_registry"
    / "exp08c_youtubevos_gtwin_d3comp_fullloss_davis_s1s2_2000"
    / "code"
    / "prepare_gtwin_manifest.py"
)

runpy.run_path(str(TARGET), run_name="__main__")
