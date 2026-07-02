"""Run the official VOID Kubric script under Blender Python.

The official script is left unmodified. This launcher only makes the isolated
Kubric/TensorFlow environment visible to Blender's embedded Python and replaces
``sys.argv`` so the script's parser does not consume Blender's own CLI flags.
"""

from __future__ import annotations

import os
import runpy
import sys


DEFAULT_SCRIPT = (
    "/mnt/nas/hj/H20_Video_inpainting_DPO/third_party/VOID/"
    "Netflix_void-model/data_generation/kubric_variable_objects.py"
)
DEFAULT_KUBRIC_SRC = "/home/hj/tools/void_kubric_exp58b/kubric_src"
DEFAULT_SITE_PACKAGES = (
    "/home/hj/conda_envs/void_kubric_exp58b/lib/python3.10/site-packages"
)


def _script_args() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return sys.argv[1:]


def main() -> None:
    kubric_src = os.environ.get("EXP58B_KUBRIC_SRC", DEFAULT_KUBRIC_SRC)
    site_packages = os.environ.get("EXP58B_SITE_PACKAGES", DEFAULT_SITE_PACKAGES)
    official_script = os.environ.get("EXP58B_OFFICIAL_SCRIPT", DEFAULT_SCRIPT)

    sys.path.insert(0, kubric_src)
    sys.path.insert(1, site_packages)

    args = _script_args()
    print("EXP58B_LAUNCH_OFFICIAL_SCRIPT", official_script)
    print("EXP58B_LAUNCH_ARGS", args)
    sys.argv = [official_script] + args
    runpy.run_path(official_script, run_name="__main__")


if __name__ == "__main__":
    main()
