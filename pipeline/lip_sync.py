from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

AVATAR_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PADS = (0, 10, 0, 0)
DEFAULT_FACE_DET_BATCH_SIZE = 4
DEFAULT_WAV2LIP_BATCH_SIZE = 16
DEFAULT_RESIZE_FACTOR = 1
DEFAULT_FPS = 25.0
SUPPORTED_CHECKPOINT_NAMES = (
    "Wav2Lip-SD-NOGAN.pt",
    "wav2lip.pth",
    "Wav2Lip-SD-GAN.pt",
    "wav2lip_gan.pth",
)
WAV2LIP_REQUIRED_MODULES = ("cv2", "librosa", "numpy", "torch")


@dataclass(frozen=True)
class Wav2LipConfig:
    wav2lip_dir: Path
    checkpoint_path: Path
    python_executable: Path
    pads: tuple[int, int, int, int] = DEFAULT_PADS
    face_det_batch_size: int = DEFAULT_FACE_DET_BATCH_SIZE
    wav2lip_batch_size: int = DEFAULT_WAV2LIP_BATCH_SIZE
    resize_factor: int = DEFAULT_RESIZE_FACTOR
    fps: float = DEFAULT_FPS
    nosmooth: bool = False
    keep_intermediate: bool = False


def _resolve_executable(candidate: Union[Path, str]) -> Optional[Path]:
    candidate_text = str(candidate)
    direct_path = Path(candidate_text).expanduser()
    if direct_path.is_file():
        return direct_path.resolve()

    discovered_path = shutil.which(candidate_text)
    if discovered_path:
        return Path(discovered_path).resolve()

    return None


def _candidate_python_paths(wav2lip_dir: Path) -> list[Path]:
    home_dir = Path.home()
    return [
        wav2lip_dir / ".venv" / "bin" / "python",
        wav2lip_dir / "venv" / "bin" / "python",
        wav2lip_dir / "env" / "bin" / "python",
        home_dir / "miniconda3" / "envs" / "wav2lip" / "bin" / "python",
        home_dir / "anaconda3" / "envs" / "wav2lip" / "bin" / "python",
        home_dir / ".conda" / "envs" / "wav2lip" / "bin" / "python",
    ]


def _python_supports_wav2lip(python_executable: Path) -> bool:
    module_check = (
        "import importlib.util, sys; "
        f"mods={WAV2LIP_REQUIRED_MODULES!r}; "
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
    wav2lip_dir: Path,
) -> Path:
    candidates: list[Union[Path, str]] = []
    if explicit_path is not None:
        candidates.append(explicit_path)

    python_env_value = os.environ.get("WAV2LIP_PYTHON")
    if python_env_value:
        candidates.append(python_env_value)

    candidates.extend(_candidate_python_paths(wav2lip_dir))

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
            if _python_supports_wav2lip(resolved_candidate):
                return resolved_candidate

    if fallback_candidate is not None:
        missing_modules = ", ".join(WAV2LIP_REQUIRED_MODULES)
        raise RuntimeError(
            "Could not find a Python interpreter with the modules Wav2Lip needs "
            f"({missing_modules}). Set WAV2LIP_PYTHON to the Wav2Lip environment "
            "or install the dependencies into a discovered local environment."
        )

    raise FileNotFoundError(
        "Could not locate a Python executable for Wav2Lip. "
        "Set WAV2LIP_PYTHON or install Wav2Lip into a discovered local environment."
    )


def _find_wav2lip_dir(explicit_path: Optional[Union[Path, str]]) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path).expanduser())

    wav2lip_env_value = os.environ.get("WAV2LIP_DIR")
    if wav2lip_env_value:
        candidates.append(Path(wav2lip_env_value).expanduser())

    candidates.append(AVATAR_ROOT.parent / "Wav2Lip")

    for candidate in candidates:
        resolved_candidate = candidate.resolve()
        if (resolved_candidate / "inference.py").is_file():
            return resolved_candidate

    raise FileNotFoundError(
        "Could not locate the Wav2Lip checkout. "
        "Set WAV2LIP_DIR or place the checkout at ../Wav2Lip."
    )


def _find_checkpoint_path(
    explicit_path: Optional[Union[Path, str]],
    wav2lip_dir: Path,
) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path).expanduser())

    checkpoint_env_value = os.environ.get("WAV2LIP_CHECKPOINT")
    if checkpoint_env_value:
        candidates.append(Path(checkpoint_env_value).expanduser())

    checkpoints_dir = wav2lip_dir / "checkpoints"
    candidates.extend(checkpoints_dir / filename for filename in SUPPORTED_CHECKPOINT_NAMES)

    for candidate in candidates:
        resolved_candidate = candidate.resolve()
        if resolved_candidate.is_file():
            return resolved_candidate

    supported_names = ", ".join(SUPPORTED_CHECKPOINT_NAMES)
    raise FileNotFoundError(
        "Could not locate a Wav2Lip checkpoint. "
        f"Set WAV2LIP_CHECKPOINT or place one of "
        f"{supported_names} inside {(wav2lip_dir / 'checkpoints').resolve()}."
    )


def _build_inference_command(
    config: Wav2LipConfig,
    source_media: Path,
    audio_path: Path,
    output_path: Path,
) -> list[str]:
    inference_script = config.wav2lip_dir / "inference.py"
    command = [
        str(config.python_executable),
        str(inference_script),
        "--checkpoint_path",
        str(config.checkpoint_path),
        "--face",
        str(source_media),
        "--audio",
        str(audio_path),
        "--outfile",
        str(output_path),
        "--fps",
        str(config.fps),
        "--pads",
        *(str(value) for value in config.pads),
        "--face_det_batch_size",
        str(config.face_det_batch_size),
        "--wav2lip_batch_size",
        str(config.wav2lip_batch_size),
        "--resize_factor",
        str(config.resize_factor),
    ]

    if config.nosmooth:
        command.append("--nosmooth")

    return command


def _build_environment(config: Wav2LipConfig) -> dict[str, str]:
    env = os.environ.copy()
    env["WAV2LIP_DIR"] = str(config.wav2lip_dir)
    env["WAV2LIP_CHECKPOINT"] = str(config.checkpoint_path)
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = env.get("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = env.get(
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1"
    )
    env["PATH"] = os.pathsep.join(
        [
            str(config.python_executable.parent),
            env.get("PATH", ""),
        ]
    )

    existing_pythonpath = env.get("PYTHONPATH")
    pythonpath_entries = [str(config.wav2lip_dir)]
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    return env


def _prepare_checkpoint(
    config: Wav2LipConfig,
    workspace_dir: Path,
    env: dict[str, str],
) -> Path:
    if config.checkpoint_path.suffix.lower() != ".pt":
        return config.checkpoint_path

    converted_checkpoint = workspace_dir / f"{config.checkpoint_path.stem}_converted.pth"
    converter_script = (
        "import sys, torch; "
        "source_path=sys.argv[1]; "
        "target_path=sys.argv[2]; "
        "scripted_model=torch.jit.load(source_path, map_location='cpu'); "
        "torch.save({'state_dict': scripted_model.state_dict()}, target_path)"
    )
    subprocess.run(
        [
            str(config.python_executable),
            "-c",
            converter_script,
            str(config.checkpoint_path),
            str(converted_checkpoint),
        ],
        cwd=config.wav2lip_dir,
        check=True,
        env=env,
    )
    return converted_checkpoint


def _run_command(command: Sequence[str], cwd: Path, env: dict[str, str]) -> None:
    try:
        subprocess.run(command, cwd=cwd, check=True, env=env)
    except FileNotFoundError as exc:
        rendered_command = " ".join(str(part) for part in command)
        raise RuntimeError(f"Failed to launch Wav2Lip command: {rendered_command}") from exc
    except subprocess.CalledProcessError as exc:
        rendered_command = " ".join(str(part) for part in command)
        raise RuntimeError(f"Wav2Lip command failed: {rendered_command}") from exc


def generate_lip_sync(
    source_media_path: Union[Path, str],
    audio_path: Union[Path, str],
    output_path: Union[Path, str],
    *,
    wav2lip_dir: Optional[Union[Path, str]] = None,
    wav2lip_python: Optional[Union[Path, str]] = None,
    wav2lip_checkpoint: Optional[Union[Path, str]] = None,
    pads: Sequence[int] = DEFAULT_PADS,
    face_det_batch_size: int = DEFAULT_FACE_DET_BATCH_SIZE,
    wav2lip_batch_size: int = DEFAULT_WAV2LIP_BATCH_SIZE,
    resize_factor: int = DEFAULT_RESIZE_FACTOR,
    fps: float = DEFAULT_FPS,
    nosmooth: bool = False,
    keep_intermediate: bool = False,
) -> Path:
    source_media = Path(source_media_path).expanduser().resolve()
    driven_audio = Path(audio_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()

    if not source_media.is_file():
        raise FileNotFoundError(f"Wav2Lip source media was not found: {source_media}")
    if not driven_audio.is_file():
        raise FileNotFoundError(f"Wav2Lip audio input was not found: {driven_audio}")

    pads_tuple = tuple(int(value) for value in pads)
    if len(pads_tuple) != 4:
        raise ValueError("Wav2Lip pads must contain exactly four integers.")

    resolved_wav2lip_dir = _find_wav2lip_dir(wav2lip_dir)
    resolved_checkpoint_path = _find_checkpoint_path(wav2lip_checkpoint, resolved_wav2lip_dir)

    config = Wav2LipConfig(
        wav2lip_dir=resolved_wav2lip_dir,
        checkpoint_path=resolved_checkpoint_path,
        python_executable=_find_python_executable(wav2lip_python, resolved_wav2lip_dir),
        pads=pads_tuple,
        face_det_batch_size=face_det_batch_size,
        wav2lip_batch_size=wav2lip_batch_size,
        resize_factor=resize_factor,
        fps=fps,
        nosmooth=nosmooth,
        keep_intermediate=keep_intermediate,
    )
    print(f"Using Wav2Lip Python: {config.python_executable}")
    print(f"Using Wav2Lip checkpoint: {config.checkpoint_path}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    workspace_dir = Path(
        tempfile.mkdtemp(prefix=f"{output_file.stem}_wav2lip_", dir=output_file.parent)
    )
    staged_source_media = workspace_dir / f"face{source_media.suffix.lower()}"
    staged_audio_path = workspace_dir / f"audio{driven_audio.suffix.lower()}"
    result_dir = workspace_dir / "results"
    temp_dir = workspace_dir / "temp"
    result_file = result_dir / output_file.name
    shutil.copy2(source_media, staged_source_media)
    shutil.copy2(driven_audio, staged_audio_path)
    result_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    env = _build_environment(config)
    numba_cache_dir = Path(env.get("NUMBA_CACHE_DIR", Path(tempfile.gettempdir()) / "avatar-numba-cache"))
    numba_cache_dir.mkdir(parents=True, exist_ok=True)
    env["NUMBA_CACHE_DIR"] = str(numba_cache_dir)
    prepared_checkpoint = _prepare_checkpoint(config, workspace_dir, env)

    command = _build_inference_command(
        config=config,
        source_media=staged_source_media,
        audio_path=staged_audio_path,
        output_path=result_file,
    )
    checkpoint_index = command.index("--checkpoint_path") + 1
    command[checkpoint_index] = str(prepared_checkpoint)

    try:
        _run_command(command, cwd=workspace_dir, env=env)
        if not result_file.is_file():
            raise FileNotFoundError(
                f"Wav2Lip finished without producing an MP4 at {result_file}."
            )
        shutil.copy2(result_file, output_file)
        return output_file
    finally:
        if not config.keep_intermediate:
            shutil.rmtree(workspace_dir, ignore_errors=True)
