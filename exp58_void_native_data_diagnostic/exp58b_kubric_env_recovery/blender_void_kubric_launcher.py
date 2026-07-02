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


def _install_openexr_channel_shim() -> None:
    """Make Kubric EXR postprocessing tolerate Blender's channel casing.

    Blender 3.6 writes cryptomatte channels as ``CryptoObject00.R`` etc. Some
    Kubric code paths request ``CryptoObject00.r``. OpenEXR 3.2 is strict about
    channel names, so this compatibility shim resolves channels
    case-insensitively while preserving the original Kubric layer semantics.
    """

    if os.environ.get("EXP58B_OPENEXR_CHANNEL_SHIM", "1") == "0":
        return

    import OpenEXR  # pylint: disable=import-outside-toplevel
    import Imath  # pylint: disable=import-outside-toplevel
    import numpy as np  # pylint: disable=import-outside-toplevel
    from kubric.renderer import blender_utils  # pylint: disable=import-outside-toplevel

    def _resolve_channel(channels_header, channel_name: str) -> str:
        if channel_name in channels_header:
            return channel_name
        lower_map = {name.lower(): name for name in channels_header}
        resolved = lower_map.get(channel_name.lower())
        if resolved is None:
            raise KeyError(channel_name)
        return resolved

    def _read_channels_from_exr(exr: OpenEXR.InputFile, channel_names):
        channels_header = exr.header()["channels"]
        window = exr.header()["dataWindow"]
        width = window.max.x - window.min.x + 1
        height = window.max.y - window.min.y + 1
        outputs = []
        for requested_name in channel_names:
            channel_name = _resolve_channel(channels_header, requested_name)
            channel_type = channels_header[channel_name].type.v
            numpy_type = {
                Imath.PixelType.HALF: np.float16,
                Imath.PixelType.FLOAT: np.float32,
                Imath.PixelType.UINT: np.uint32,
            }[channel_type]
            array = np.frombuffer(exr.channel(channel_name), numpy_type)
            array = array.reshape([height, width])
            outputs.append(array)
        return np.stack(outputs, axis=-1)

    def _get_render_layers_from_exr(filename):
        exr = OpenEXR.InputFile(str(filename))
        layer_names = set()
        for channel_name in exr.header()["channels"]:
            layer_name, _, _ = channel_name.partition(".")
            layer_names.add(layer_name)

        output = {}
        if "Image" in layer_names:
            output["linear_rgba"] = _read_channels_from_exr(
                exr, ["Image.R", "Image.G", "Image.B", "Image.A"]
            )
        if "Depth" in layer_names:
            output["depth"] = _read_channels_from_exr(exr, ["Depth.V"])
        if "Vector" in layer_names:
            flow = _read_channels_from_exr(
                exr, ["Vector.R", "Vector.G", "Vector.B", "Vector.A"]
            )
            output["backward_flow"] = np.zeros_like(flow[..., :2])
            output["backward_flow"][..., 0] = flow[..., 1]
            output["backward_flow"][..., 1] = -flow[..., 0]
            output["forward_flow"] = np.zeros_like(flow[..., 2:])
            output["forward_flow"][..., 0] = flow[..., 3]
            output["forward_flow"][..., 1] = -flow[..., 2]
        if "Normal" in layer_names:
            output["normal"] = _read_channels_from_exr(
                exr, ["Normal.X", "Normal.Y", "Normal.Z"]
            )
        if "UV" in layer_names:
            output["uv"] = _read_channels_from_exr(exr, ["UV.X", "UV.Y", "UV.Z"])
        if "CryptoObject00" in layer_names:
            crypto_layers = [name for name in layer_names if name.startswith("CryptoObject")]
            index_channels = [name + "." + c for name in crypto_layers for c in "rb"]
            idxs = _read_channels_from_exr(exr, index_channels)
            idxs.dtype = np.uint32
            output["segmentation_indices"] = idxs
            alpha_channels = [name + "." + c for name in crypto_layers for c in "ga"]
            output["segmentation_alphas"] = _read_channels_from_exr(exr, alpha_channels)
        if "ObjectCoordinates" in layer_names:
            output["object_coordinates"] = _read_channels_from_exr(
                exr,
                ["ObjectCoordinates.R", "ObjectCoordinates.G", "ObjectCoordinates.B"],
            )
        return output

    blender_utils.get_render_layers_from_exr = _get_render_layers_from_exr


def main() -> None:
    kubric_src = os.environ.get("EXP58B_KUBRIC_SRC", DEFAULT_KUBRIC_SRC)
    site_packages = os.environ.get("EXP58B_SITE_PACKAGES", DEFAULT_SITE_PACKAGES)
    official_script = os.environ.get("EXP58B_OFFICIAL_SCRIPT", DEFAULT_SCRIPT)

    sys.path.insert(0, kubric_src)
    sys.path.insert(1, site_packages)
    _install_openexr_channel_shim()

    args = _script_args()
    print("EXP58B_LAUNCH_OFFICIAL_SCRIPT", official_script)
    print("EXP58B_LAUNCH_ARGS", args)
    sys.argv = [official_script] + args
    runpy.run_path(official_script, run_name="__main__")


if __name__ == "__main__":
    main()
