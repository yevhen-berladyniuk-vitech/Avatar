import os
import runpy
import sys

import face_alignment
import torch

INFERENCE_PATH = "/app/SadTalker/inference.py"

_ORIGINAL_FACE_ALIGNMENT = face_alignment.FaceAlignment
_ORIGINAL_TORCH_LOAD = torch.load


def _patched_face_alignment(*args, **kwargs):
    kwargs.setdefault("device", os.environ.get("SADTALKER_FACE_ALIGNMENT_DEVICE", "cuda"))
    return _ORIGINAL_FACE_ALIGNMENT(*args, **kwargs)


def _patched_torch_load(*args, **kwargs):
    if os.environ.get("SADTALKER_FORCE_CPU_LOAD") == "1":
        kwargs.setdefault("map_location", torch.device("cpu"))
    return _ORIGINAL_TORCH_LOAD(*args, **kwargs)


def main() -> None:
    if "--cpu" in sys.argv[1:]:
        os.environ.setdefault("SADTALKER_FACE_ALIGNMENT_DEVICE", "cpu")
        os.environ.setdefault("SADTALKER_FORCE_CPU_LOAD", "1")

    face_alignment.FaceAlignment = _patched_face_alignment
    torch.load = _patched_torch_load
    sys.argv[0] = INFERENCE_PATH
    runpy.run_path(INFERENCE_PATH, run_name="__main__")


if __name__ == "__main__":
    main()
