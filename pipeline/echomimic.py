from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union
from uuid import uuid4

AVATAR_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTEXT_FRAMES = 12
DEFAULT_CONTEXT_OVERLAP = 3
DEFAULT_DEVICE = "cpu"
DEFAULT_FACEMASK_DILATION_RATIO = 0.1
DEFAULT_FACECROP_DILATION_RATIO = 0.5
DEFAULT_FPS = 24
DEFAULT_HEIGHT = 512
DEFAULT_LENGTH = 1200
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_SEED = 420
DEFAULT_WIDTH = 512
ECHOMIMIC_REQUIRED_MODULES = (
    "diffusers",
    "numpy",
    "omegaconf",
    "PIL",
    "torch",
)
STATE_DICT_CONVERTER_REQUIRED_MODULES = (
    "safetensors",
    "torch",
)
IMAGE_SOURCE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
VIDEO_SOURCE_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4"}

VARIANT_SETTINGS = {
    "standard": {
        "default_cfg": 2.5,
        "default_steps": 30,
        "denoising_checkpoint": "denoising_unet.pth",
        "face_locator_checkpoint": "face_locator.pth",
        "motion_module_checkpoint": "motion_module.pth",
        "reference_checkpoint": "reference_unet.pth",
    },
    "accelerated": {
        "default_cfg": 1.0,
        "default_steps": 6,
        "denoising_checkpoint": "denoising_unet_acc.pth",
        "face_locator_checkpoint": "face_locator.pth",
        "motion_module_checkpoint": "motion_module_acc.pth",
        "reference_checkpoint": "reference_unet.pth",
    },
}


@dataclass(frozen=True)
class EchoMimicConfig:
    echomimic_dir: Path
    python_executable: Path
    script_path: Path
    variant: str
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    length: int = DEFAULT_LENGTH
    seed: int = DEFAULT_SEED
    facemask_dilation_ratio: float = DEFAULT_FACEMASK_DILATION_RATIO
    facecrop_dilation_ratio: float = DEFAULT_FACECROP_DILATION_RATIO
    context_frames: int = DEFAULT_CONTEXT_FRAMES
    context_overlap: int = DEFAULT_CONTEXT_OVERLAP
    cfg: float = 1.0
    steps: int = 6
    sample_rate: int = DEFAULT_SAMPLE_RATE
    fps: int = DEFAULT_FPS
    device: str = DEFAULT_DEVICE
    keep_intermediate: bool = False


def _resolve_executable(candidate: Union[Path, str]) -> Optional[Path]:
    candidate_text = str(candidate)
    direct_path = Path(candidate_text).expanduser()
    if direct_path.is_file():
        return direct_path

    discovered_path = shutil.which(candidate_text)
    if discovered_path:
        return Path(discovered_path)

    return None


def _candidate_python_paths(echomimic_dir: Path) -> list[Path]:
    home_dir = Path.home()
    return [
        AVATAR_ROOT / ".venv-echomimic311" / "bin" / "python",
        echomimic_dir / ".venv" / "bin" / "python",
        echomimic_dir / "venv" / "bin" / "python",
        echomimic_dir / "env" / "bin" / "python",
        home_dir / "miniconda3" / "envs" / "echomimic" / "bin" / "python",
        home_dir / "anaconda3" / "envs" / "echomimic" / "bin" / "python",
        home_dir / ".conda" / "envs" / "echomimic" / "bin" / "python",
    ]


def _candidate_ffmpeg_paths(python_executable: Path) -> list[Union[Path, str]]:
    home_dir = Path.home()
    candidates: list[Union[Path, str]] = []

    ffmpeg_env_value = os.environ.get("FFMPEG")
    if ffmpeg_env_value:
        candidates.append(ffmpeg_env_value)

    ffmpeg_path_env_value = os.environ.get("FFMPEG_PATH")
    if ffmpeg_path_env_value:
        ffmpeg_path = Path(ffmpeg_path_env_value).expanduser()
        if ffmpeg_path.is_dir():
            candidates.append(ffmpeg_path / "ffmpeg")
        else:
            candidates.append(ffmpeg_path)

    candidates.extend(
        [
            python_executable.parent / "ffmpeg",
            home_dir / "miniconda3" / "envs" / "echomimic" / "bin" / "ffmpeg",
            home_dir / "anaconda3" / "envs" / "echomimic" / "bin" / "ffmpeg",
            home_dir / ".conda" / "envs" / "echomimic" / "bin" / "ffmpeg",
            "ffmpeg",
        ]
    )
    return candidates


def _find_ffmpeg_executable(python_executable: Path) -> Optional[Path]:
    for candidate in _candidate_ffmpeg_paths(python_executable):
        resolved_candidate = _resolve_executable(candidate)
        if resolved_candidate is not None:
            return resolved_candidate
    return None


def _python_supports_echomimic(python_executable: Path) -> bool:
    module_check = (
        "import importlib.util, sys; "
        f"mods={ECHOMIMIC_REQUIRED_MODULES!r}; "
        "missing=[m for m in mods if importlib.util.find_spec(m) is None]; "
        "sys.exit(0 if not missing else 1)"
    )
    try:
        subprocess.run(
            [str(python_executable), "-c", module_check],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return True


def _python_supports_modules(
    python_executable: Path,
    modules: Sequence[str],
) -> bool:
    module_check = (
        "import importlib.util, sys; "
        f"mods={tuple(modules)!r}; "
        "missing=[m for m in mods if importlib.util.find_spec(m) is None]; "
        "sys.exit(0 if not missing else 1)"
    )
    try:
        subprocess.run(
            [str(python_executable), "-c", module_check],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return True


def _find_python_executable(
    explicit_path: Optional[Union[Path, str]],
    echomimic_dir: Path,
) -> Path:
    candidates: list[Union[Path, str]] = []
    if explicit_path is not None:
        candidates.append(explicit_path)

    python_env_value = os.environ.get("ECHOMIMIC_PYTHON")
    if python_env_value:
        candidates.append(python_env_value)

    candidates.extend(_candidate_python_paths(echomimic_dir))

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "bin" / "python")

    if sys.executable:
        candidates.append(sys.executable)

    candidates.extend(["python3", "python"])

    fallback_candidate: Optional[Path] = None
    for candidate in candidates:
        resolved_candidate = _resolve_executable(candidate)
        if resolved_candidate is not None:
            if fallback_candidate is None:
                fallback_candidate = resolved_candidate
            if _python_supports_echomimic(resolved_candidate):
                return resolved_candidate

    if fallback_candidate is not None:
        missing_modules = ", ".join(ECHOMIMIC_REQUIRED_MODULES)
        raise RuntimeError(
            "Could not find a Python interpreter with the modules EchoMimic needs "
            f"({missing_modules}). Pass --echomimic-python or set ECHOMIMIC_PYTHON "
            "to the EchoMimic environment."
        )

    raise FileNotFoundError(
        "Could not locate a Python executable for EchoMimic. "
        "Pass --echomimic-python or set ECHOMIMIC_PYTHON."
    )


def _find_state_dict_converter_python(
    preferred_python: Path,
    echomimic_dir: Path,
) -> Path:
    candidates: list[Union[Path, str]] = []

    if sys.executable:
        candidates.append(sys.executable)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "bin" / "python")

    home_dir = Path.home()
    candidates.extend(
        [
            home_dir / "miniconda3" / "bin" / "python",
            home_dir / "anaconda3" / "bin" / "python",
            preferred_python,
        ]
    )
    candidates.extend(_candidate_python_paths(echomimic_dir))
    candidates.extend(["python3", "python"])

    for candidate in candidates:
        resolved_candidate = _resolve_executable(candidate)
        if resolved_candidate is not None and _python_supports_modules(
            resolved_candidate,
            STATE_DICT_CONVERTER_REQUIRED_MODULES,
        ):
            return resolved_candidate

    missing_modules = ", ".join(STATE_DICT_CONVERTER_REQUIRED_MODULES)
    raise RuntimeError(
        "Could not find a Python interpreter with the modules needed to convert "
        f"EchoMimic checkpoints to safetensors ({missing_modules})."
    )


def _find_echomimic_dir(explicit_path: Optional[Union[Path, str]]) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path).expanduser())

    echomimic_env_value = os.environ.get("ECHOMIMIC_DIR")
    if echomimic_env_value:
        candidates.append(Path(echomimic_env_value).expanduser())

    candidates.extend(
        [
            AVATAR_ROOT.parent / "EchoMimic",
            AVATAR_ROOT.parent / "echomimic",
        ]
    )

    for candidate in candidates:
        resolved_candidate = candidate.resolve()
        if (resolved_candidate / "infer_audio2vid.py").is_file():
            return resolved_candidate

    raise FileNotFoundError(
        "Could not locate the EchoMimic checkout. "
        "Pass --echomimic-dir or set ECHOMIMIC_DIR."
    )


def _is_git_lfs_pointer(file_path: Path) -> bool:
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            first_line = handle.readline().strip()
    except UnicodeDecodeError:
        return False
    return first_line == "version https://git-lfs.github.com/spec/v1"


def _validate_model_assets(echomimic_dir: Path, variant: str) -> None:
    variant_settings = VARIANT_SETTINGS[variant]
    pretrained_dir = echomimic_dir / "pretrained_weights"
    required_files = [
        pretrained_dir / variant_settings["denoising_checkpoint"],
        pretrained_dir / variant_settings["reference_checkpoint"],
        pretrained_dir / variant_settings["face_locator_checkpoint"],
        pretrained_dir / variant_settings["motion_module_checkpoint"],
        pretrained_dir / "audio_processor" / "whisper_tiny.pt",
        pretrained_dir
        / "sd-image-variations-diffusers"
        / "image_encoder"
        / "pytorch_model.bin",
        pretrained_dir
        / "sd-image-variations-diffusers"
        / "safety_checker"
        / "pytorch_model.bin",
        pretrained_dir
        / "sd-image-variations-diffusers"
        / "unet"
        / "diffusion_pytorch_model.bin",
        pretrained_dir
        / "sd-image-variations-diffusers"
        / "vae"
        / "diffusion_pytorch_model.bin",
        pretrained_dir / "sd-vae-ft-mse" / "diffusion_pytorch_model.bin",
    ]
    required_directories = [
        pretrained_dir / "sd-image-variations-diffusers",
        pretrained_dir / "sd-vae-ft-mse",
    ]

    missing_files = [path for path in required_files if not path.is_file()]
    if missing_files:
        rendered_paths = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            "EchoMimic model assets are incomplete. Missing files: "
            f"{rendered_paths}"
        )

    lfs_pointer_files = [path for path in required_files if _is_git_lfs_pointer(path)]
    if lfs_pointer_files:
        rendered_paths = ", ".join(str(path) for path in lfs_pointer_files)
        raise RuntimeError(
            "EchoMimic checkpoints have not been downloaded from Git LFS yet. "
            f"These files are still pointer stubs: {rendered_paths}. "
            "Run `git lfs pull` inside the pretrained_weights checkout."
        )

    missing_directories = [path for path in required_directories if not path.is_dir()]
    if missing_directories:
        rendered_paths = ", ".join(str(path) for path in missing_directories)
        raise FileNotFoundError(
            "EchoMimic model directories are incomplete. Missing: "
            f"{rendered_paths}"
        )


def _ensure_safetensors_checkpoint(
    checkpoint_path: Path,
    converter_python: Path,
) -> None:
    if checkpoint_path.suffix.lower() not in {".ckpt", ".pt", ".pth"}:
        return

    safetensors_path = checkpoint_path.with_suffix(".safetensors")
    if safetensors_path.is_file():
        return

    print(f"Converting EchoMimic checkpoint to safetensors: {checkpoint_path.name}")
    conversion_script = "\n".join(
        [
            "import sys",
            "from pathlib import Path",
            "import torch",
            "from safetensors.torch import save_file",
            "",
            "source_path = Path(sys.argv[1])",
            "target_path = Path(sys.argv[2])",
            "target_path.parent.mkdir(parents=True, exist_ok=True)",
            "load_kwargs = {'map_location': 'cpu'}",
            "try:",
            "    state_dict = torch.load(source_path, weights_only=True, **load_kwargs)",
            "except TypeError:",
            "    state_dict = torch.load(source_path, **load_kwargs)",
            "if not isinstance(state_dict, dict):",
            "    raise TypeError(f'Expected a state dict in {source_path}, got {type(state_dict).__name__}')",
            "tensor_state = {}",
            "for key, value in state_dict.items():",
            "    if not torch.is_tensor(value):",
            "        raise TypeError(f'Checkpoint {source_path} contains a non-tensor value at {key!r}')",
            "    tensor_state[key] = value.detach().cpu().contiguous()",
            "temp_path = target_path.with_suffix(target_path.suffix + '.tmp')",
            "save_file(tensor_state, str(temp_path))",
            "temp_path.replace(target_path)",
        ]
    )
    try:
        subprocess.run(
            [
                str(converter_python),
                "-c",
                conversion_script,
                str(checkpoint_path),
                str(safetensors_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        combined_output = (exc.stderr or "").strip()
        raise RuntimeError(
            "Failed to convert an EchoMimic checkpoint to safetensors. "
            f"Source: {checkpoint_path}. stderr: {combined_output or '<empty>'}"
        ) from exc


def _prepare_safetensors_checkpoints(
    echomimic_dir: Path,
    variant: str,
    converter_python: Path,
) -> None:
    variant_settings = VARIANT_SETTINGS[variant]
    pretrained_dir = echomimic_dir / "pretrained_weights"
    checkpoint_paths = [
        pretrained_dir / variant_settings["denoising_checkpoint"],
        pretrained_dir / variant_settings["reference_checkpoint"],
        pretrained_dir / variant_settings["face_locator_checkpoint"],
        pretrained_dir / variant_settings["motion_module_checkpoint"],
    ]
    for checkpoint_path in checkpoint_paths:
        _ensure_safetensors_checkpoint(checkpoint_path, converter_python)


def _extract_first_video_frame(source_video: Path, output_image: Path) -> Path:
    import av
    from PIL import Image

    with av.open(str(source_video)) as container:
        frame = next(container.decode(video=0), None)
        if frame is None:
            raise RuntimeError(f"Could not decode a video frame from {source_video}.")

        image = Image.fromarray(frame.to_rgb().to_ndarray())
        image.save(output_image)
        return output_image


def _stage_source_media(source_media: Path, workspace_dir: Path) -> Path:
    token = uuid4().hex[:12]
    if source_media.suffix.lower() in VIDEO_SOURCE_EXTENSIONS:
        staged_image_path = workspace_dir / f"source_{token}.png"
        return _extract_first_video_frame(source_media, staged_image_path)

    staged_media_path = workspace_dir / f"source_{token}{source_media.suffix.lower()}"
    shutil.copy2(source_media, staged_media_path)
    return staged_media_path


def _write_prompt_config(
    config_path: Path,
    echomimic_dir: Path,
    variant: str,
    source_media: Path,
    audio_path: Path,
    device: str,
) -> None:
    variant_settings = VARIANT_SETTINGS[variant]
    pretrained_dir = echomimic_dir / "pretrained_weights"
    weight_dtype = "fp16" if "cuda" in device.lower() else "fp32"
    config_text = "\n".join(
        [
            f"pretrained_base_model_path: {json.dumps(str(pretrained_dir / 'sd-image-variations-diffusers'))}",
            f"pretrained_vae_path: {json.dumps(str(pretrained_dir / 'sd-vae-ft-mse'))}",
            f"audio_model_path: {json.dumps(str(pretrained_dir / 'audio_processor' / 'whisper_tiny.pt'))}",
            "",
            f"denoising_unet_path: {json.dumps(str(pretrained_dir / variant_settings['denoising_checkpoint']))}",
            f"reference_unet_path: {json.dumps(str(pretrained_dir / variant_settings['reference_checkpoint']))}",
            f"face_locator_path: {json.dumps(str(pretrained_dir / variant_settings['face_locator_checkpoint']))}",
            f"motion_module_path: {json.dumps(str(pretrained_dir / variant_settings['motion_module_checkpoint']))}",
            "",
            f"inference_config: {json.dumps(str(echomimic_dir / 'configs' / 'inference' / 'inference_v2.yaml'))}",
            f"weight_dtype: {json.dumps(weight_dtype)}",
            "",
            "test_cases:",
            f"  {json.dumps(str(source_media))}:",
            f"    - {json.dumps(str(audio_path))}",
            "",
        ]
    )
    config_path.write_text(config_text, encoding="utf-8")


def _build_inference_command(
    config: EchoMimicConfig,
    prompt_config_path: Path,
    result_file: Path,
) -> list[str]:
    return [
        str(config.python_executable),
        str(config.script_path),
        "--echomimic-dir",
        str(config.echomimic_dir),
        "--variant",
        config.variant,
        "--config",
        str(prompt_config_path),
        "--output",
        str(result_file),
        "-W",
        str(config.width),
        "-H",
        str(config.height),
        "-L",
        str(config.length),
        "--seed",
        str(config.seed),
        "--facemusk_dilation_ratio",
        str(config.facemask_dilation_ratio),
        "--facecrop_dilation_ratio",
        str(config.facecrop_dilation_ratio),
        "--context_frames",
        str(config.context_frames),
        "--context_overlap",
        str(config.context_overlap),
        "--cfg",
        str(config.cfg),
        "--steps",
        str(config.steps),
        "--sample_rate",
        str(config.sample_rate),
        "--fps",
        str(config.fps),
        "--device",
        config.device,
    ]


def _build_environment(
    *,
    echomimic_dir: Path,
    python_executable: Path,
) -> dict[str, str]:
    env = os.environ.copy()
    env["ECHOMIMIC_DIR"] = str(echomimic_dir)
    path_entries = [str(python_executable.parent)]

    ffmpeg_executable = _find_ffmpeg_executable(python_executable)
    if ffmpeg_executable is not None:
        path_entries.append(str(ffmpeg_executable.parent))
        env["FFMPEG"] = str(ffmpeg_executable)
        env["FFMPEG_PATH"] = str(ffmpeg_executable.parent)

    path_entries.append(env.get("PATH", ""))
    env["PATH"] = os.pathsep.join(path_entries)

    existing_pythonpath = env.get("PYTHONPATH")
    pythonpath_entries = [str(echomimic_dir)]
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = env.get("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    return env


def _probe_available_device(python_executable: Path, env: dict[str, str]) -> str:
    probe_script = (
        "import torch; "
        "mps_backend=getattr(torch.backends, 'mps', None); "
        "mps_available=bool(mps_backend and mps_backend.is_available()); "
        "print('cuda' if torch.cuda.is_available() else 'mps' if mps_available else 'cpu')"
    )
    try:
        completed_process = subprocess.run(
            [str(python_executable), "-c", probe_script],
            check=True,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        combined_output = (exc.stderr or "").strip()
        raise RuntimeError(
            "The selected EchoMimic Python could not probe available devices. "
            f"stderr: {combined_output or '<empty>'}"
        ) from exc
    detected_device = completed_process.stdout.strip().lower()
    if detected_device not in {"cuda", "mps", "cpu"}:
        raise RuntimeError(
            f"Unexpected EchoMimic device probe result: {detected_device or '<empty>'}"
        )
    return detected_device


def _resolve_requested_device(
    requested_device: str,
    python_executable: Path,
    env: dict[str, str],
) -> str:
    normalized_requested_device = requested_device.strip().lower()
    if normalized_requested_device == "cpu":
        return "cpu"

    detected_device = _probe_available_device(python_executable, env)

    if normalized_requested_device == "auto":
        return detected_device

    if normalized_requested_device.startswith("cuda") and detected_device != "cuda":
        raise RuntimeError(
            "EchoMimic was asked to use CUDA, but the selected environment does not "
            "report CUDA availability on this machine. Pass `--device auto`, "
            "`--device mps`, or `--device cpu` instead."
        )
    if normalized_requested_device == "mps" and detected_device != "mps":
        raise RuntimeError(
            "EchoMimic was asked to use MPS, but the selected environment does not "
            "report MPS availability on this machine."
        )
    return requested_device


def _validate_runtime_preflight(config: EchoMimicConfig, env: dict[str, str]) -> None:
    probe_script = (
        "import diffusers; "
        "import numpy; "
        "import torch; "
        "print(torch.__version__)"
    )
    try:
        subprocess.run(
            [str(config.python_executable), "-c", probe_script],
            check=True,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        combined_output = (exc.stderr or "").strip()
        if "OMP: Error #179: Function Can't open SHM2 failed" in combined_output:
            raise RuntimeError(
                "The selected EchoMimic Python cannot import torch cleanly on this "
                "machine. The environment aborts inside the OpenMP runtime with "
                "`OMP: Error #179: Function Can't open SHM2 failed` before inference "
                "starts. Rebuild the `echomimic` environment with a macOS-compatible "
                "PyTorch/OpenMP stack, or run EchoMimic on a machine where that "
                "environment can import torch successfully."
            ) from exc
        raise RuntimeError(
            "The selected EchoMimic Python failed a runtime preflight import check. "
            f"stderr: {combined_output or '<empty>'}"
        ) from exc
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Failed to launch the EchoMimic Python preflight: {config.python_executable}"
        ) from exc


def _run_command(command: Sequence[str], cwd: Path, env: dict[str, str]) -> None:
    try:
        subprocess.run(command, cwd=cwd, check=True, env=env)
    except FileNotFoundError as exc:
        rendered_command = " ".join(str(part) for part in command)
        raise RuntimeError(
            f"Failed to launch EchoMimic command: {rendered_command}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        rendered_command = " ".join(str(part) for part in command)
        raise RuntimeError(
            f"EchoMimic command failed: {rendered_command}"
        ) from exc


def generate_echo_mimic(
    source_media_path: Union[Path, str],
    audio_path: Union[Path, str],
    output_path: Union[Path, str],
    *,
    echomimic_dir: Optional[Union[Path, str]] = None,
    echomimic_python: Optional[Union[Path, str]] = None,
    variant: str = "accelerated",
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    length: int = DEFAULT_LENGTH,
    seed: int = DEFAULT_SEED,
    facemask_dilation_ratio: float = DEFAULT_FACEMASK_DILATION_RATIO,
    facecrop_dilation_ratio: float = DEFAULT_FACECROP_DILATION_RATIO,
    context_frames: int = DEFAULT_CONTEXT_FRAMES,
    context_overlap: int = DEFAULT_CONTEXT_OVERLAP,
    cfg: Optional[float] = None,
    steps: Optional[int] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    fps: int = DEFAULT_FPS,
    device: str = DEFAULT_DEVICE,
    keep_intermediate: bool = False,
) -> Path:
    source_media = Path(source_media_path).expanduser().resolve()
    driven_audio = Path(audio_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()

    if variant not in VARIANT_SETTINGS:
        supported_variants = ", ".join(sorted(VARIANT_SETTINGS))
        raise ValueError(
            f"Unsupported EchoMimic variant '{variant}'. Use one of: {supported_variants}"
        )

    if not source_media.is_file():
        raise FileNotFoundError(f"EchoMimic source media was not found: {source_media}")
    if source_media.suffix.lower() not in IMAGE_SOURCE_EXTENSIONS | VIDEO_SOURCE_EXTENSIONS:
        supported_extensions = ", ".join(
            sorted(IMAGE_SOURCE_EXTENSIONS | VIDEO_SOURCE_EXTENSIONS)
        )
        raise ValueError(
            f"Unsupported EchoMimic source media type for {source_media}. "
            f"Use one of: {supported_extensions}"
        )
    if not driven_audio.is_file():
        raise FileNotFoundError(f"EchoMimic audio input was not found: {driven_audio}")

    resolved_echomimic_dir = _find_echomimic_dir(echomimic_dir)
    _validate_model_assets(resolved_echomimic_dir, variant)
    resolved_python_executable = _find_python_executable(
        echomimic_python, resolved_echomimic_dir
    )
    converter_python = _find_state_dict_converter_python(
        resolved_python_executable,
        resolved_echomimic_dir,
    )
    _prepare_safetensors_checkpoints(
        resolved_echomimic_dir,
        variant,
        converter_python,
    )
    runner_path = (AVATAR_ROOT / "pipeline" / "echomimic_runner.py").resolve()
    if not runner_path.is_file():
        raise FileNotFoundError(f"Avatar EchoMimic runner was not found: {runner_path}")

    probe_env = _build_environment(
        echomimic_dir=resolved_echomimic_dir,
        python_executable=resolved_python_executable,
    )
    resolved_device = _resolve_requested_device(
        device,
        resolved_python_executable,
        probe_env,
    )
    variant_settings = VARIANT_SETTINGS[variant]

    config = EchoMimicConfig(
        echomimic_dir=resolved_echomimic_dir,
        python_executable=resolved_python_executable,
        script_path=runner_path,
        variant=variant,
        width=width,
        height=height,
        length=length,
        seed=seed,
        facemask_dilation_ratio=facemask_dilation_ratio,
        facecrop_dilation_ratio=facecrop_dilation_ratio,
        context_frames=context_frames,
        context_overlap=context_overlap,
        cfg=cfg if cfg is not None else float(variant_settings["default_cfg"]),
        steps=steps if steps is not None else int(variant_settings["default_steps"]),
        sample_rate=sample_rate,
        fps=fps,
        device=resolved_device,
        keep_intermediate=keep_intermediate,
    )
    print(f"Using EchoMimic Python: {config.python_executable}")
    print(f"Using EchoMimic variant: {config.variant}")
    print(f"Using EchoMimic device: {config.device}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    workspace_dir = Path(
        tempfile.mkdtemp(prefix=f"{output_file.stem}_echomimic_", dir=output_file.parent)
    )
    staged_audio_path = workspace_dir / f"audio_{uuid4().hex[:12]}{driven_audio.suffix.lower()}"
    shutil.copy2(driven_audio, staged_audio_path)
    env = _build_environment(
        echomimic_dir=config.echomimic_dir,
        python_executable=config.python_executable,
    )
    _validate_runtime_preflight(config, env)

    try:
        staged_source_media = _stage_source_media(source_media, workspace_dir)
        prompt_config_path = workspace_dir / "animation_avatar.yaml"
        result_file = workspace_dir / "echomimic_silent.mp4"
        _write_prompt_config(
            config_path=prompt_config_path,
            echomimic_dir=config.echomimic_dir,
            variant=config.variant,
            source_media=staged_source_media,
            audio_path=staged_audio_path,
            device=config.device,
        )
        command = _build_inference_command(config, prompt_config_path, result_file)
        _run_command(command, cwd=workspace_dir, env=env)
        if not result_file.is_file():
            raise FileNotFoundError(
                f"EchoMimic runner finished without producing {result_file}."
            )
        shutil.copy2(result_file, output_file)
        return output_file
    finally:
        if not config.keep_intermediate:
            shutil.rmtree(workspace_dir, ignore_errors=True)
