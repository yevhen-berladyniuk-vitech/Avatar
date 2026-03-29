from __future__ import annotations

import fractions
import importlib.util
import shutil
import tempfile
from pathlib import Path
from typing import Final, Union

RVM_REPOSITORY: Final = "PeterL1n/RobustVideoMatting"
RVM_MODEL_VARIANT: Final = "mobilenetv3"
DEFAULT_OUTPUT_VIDEO_MBPS: Final = 4
DEFAULT_FRAME_RATE: Final = 25
VIDEO_SOURCE_EXTENSIONS: Final = {".avi", ".m4v", ".mkv", ".mov", ".mp4"}
REQUIRED_MODULE_PACKAGES: Final = {
    "av": "av",
    "numpy": "numpy",
    "torch": "torch",
    "torchvision": "torchvision",
}


def _validate_source_video(source_video: Path) -> None:
    if not source_video.is_file():
        raise FileNotFoundError(
            f"RobustVideoMatting source video was not found: {source_video}"
        )

    if source_video.suffix.lower() not in VIDEO_SOURCE_EXTENSIONS:
        supported_extensions = ", ".join(sorted(VIDEO_SOURCE_EXTENSIONS))
        raise ValueError(
            "RobustVideoMatting requires a video input. "
            f"Use one of: {supported_extensions}"
        )


def _validate_runtime_dependencies() -> None:
    missing_modules = [
        module_name
        for module_name in REQUIRED_MODULE_PACKAGES
        if importlib.util.find_spec(module_name) is None
    ]
    if missing_modules:
        rendered_packages = ", ".join(
            REQUIRED_MODULE_PACKAGES[module_name] for module_name in missing_modules
        )
        raise RuntimeError(
            "RobustVideoMatting requires additional Python packages in the current "
            f"environment. Install: {rendered_packages}"
        )


def _detect_device(torch_module: object) -> str:
    if torch_module.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch_module.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"


def _auto_downsample_ratio(height: int, width: int) -> float:
    return min(512 / max(height, width), 1)


def _load_model(torch_module: object, device: str) -> object:
    try:
        model = torch_module.hub.load(RVM_REPOSITORY, RVM_MODEL_VARIANT)
    except Exception as exc:
        raise RuntimeError(
            "Could not load RobustVideoMatting from TorchHub. "
            "Make sure the machine has internet access on the first run."
        ) from exc

    return model.eval().to(device)


def _resolve_frame_rate(video_stream: object) -> fractions.Fraction:
    average_rate = getattr(video_stream, "average_rate", None)
    if average_rate is not None:
        return average_rate
    return fractions.Fraction(DEFAULT_FRAME_RATE, 1)


def generate_background_video(
    source_media_path: Union[Path, str],
    output_path: Union[Path, str],
) -> Path:
    source_video = Path(source_media_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()

    _validate_source_video(source_video)
    _validate_runtime_dependencies()

    import av
    import numpy as np
    import torch

    device = _detect_device(torch)
    print(f"Using RobustVideoMatting via TorchHub on device: {device}")

    model = _load_model(torch, device)
    model_parameter = next(model.parameters())
    model_dtype = model_parameter.dtype

    output_file.parent.mkdir(parents=True, exist_ok=True)
    workspace_dir = Path(
        tempfile.mkdtemp(prefix=f"{output_file.stem}_rvm_", dir=output_file.parent)
    )
    staged_output_path = workspace_dir / output_file.name

    try:
        with av.open(str(source_video)) as input_container:
            with av.open(str(staged_output_path), mode="w") as output_container:
                with torch.no_grad():
                    input_stream = next(
                        (stream for stream in input_container.streams if stream.type == "video"),
                        None,
                    )
                    if input_stream is None:
                        raise RuntimeError(f"No video stream found in {source_video}.")

                    output_stream = output_container.add_stream(
                        "mpeg4",
                        rate=_resolve_frame_rate(input_stream),
                    )
                    output_stream.width = input_stream.codec_context.width
                    output_stream.height = input_stream.codec_context.height
                    output_stream.bit_rate = int(DEFAULT_OUTPUT_VIDEO_MBPS * 1_000_000)
                    output_stream.pix_fmt = "yuv420p"

                    recurrence = [None] * 4
                    downsample_ratio: float | None = None

                    for frame in input_container.decode(video=0):
                        frame_rgb = frame.to_rgb().to_ndarray()
                        frame_tensor = (
                            torch.from_numpy(frame_rgb)
                            .permute(2, 0, 1)
                            .unsqueeze(0)
                            .to(device=device, dtype=model_dtype)
                            .div(255.0)
                            .unsqueeze(0)
                        )
                        if downsample_ratio is None:
                            _, _, _, height, width = frame_tensor.shape
                            downsample_ratio = _auto_downsample_ratio(height, width)

                        foreground, alpha, *recurrence = model(
                            frame_tensor,
                            *recurrence,
                            downsample_ratio,
                        )
                        composition = foreground * alpha
                        output_array = (
                            composition[0, 0]
                            .permute(1, 2, 0)
                            .clamp(0, 1)
                            .mul(255)
                            .byte()
                            .cpu()
                            .numpy()
                        )
                        output_frame = av.VideoFrame.from_ndarray(
                            np.ascontiguousarray(output_array),
                            format="rgb24",
                        )
                        for packet in output_stream.encode(output_frame):
                            output_container.mux(packet)

                    for packet in output_stream.encode():
                        output_container.mux(packet)

        if not staged_output_path.is_file():
            raise FileNotFoundError(
                "RobustVideoMatting finished without producing the expected output "
                f"video: {staged_output_path}"
            )
        shutil.copy2(staged_output_path, output_file)
        return output_file
    finally:
        shutil.rmtree(workspace_dir, ignore_errors=True)
