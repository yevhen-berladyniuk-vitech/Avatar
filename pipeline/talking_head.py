import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

AVATAR_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SADTALKER_ENV_NAME = "sadtalker"


def _default_sadtalker_dir() -> Path:
    configured_dir = os.environ.get("SADTALKER_DIR")
    if configured_dir:
        return Path(configured_dir).expanduser()
    return AVATAR_ROOT.parent / "SadTalker"


def _default_checkpoint_dir(sadtalker_dir: Path) -> Path:
    configured_dir = os.environ.get("SADTALKER_CHECKPOINT_DIR")
    if configured_dir:
        return Path(configured_dir).expanduser()
    return sadtalker_dir / "checkpoints"


@dataclass(frozen=True)
class SadTalkerConfig:
    sadtalker_dir: Path
    checkpoint_dir: Path
    conda_executable: Path
    conda_env_name: str = DEFAULT_SADTALKER_ENV_NAME
    preprocess: str = "crop"
    size: int = 256
    pose_style: int = 0
    batch_size: int = 2
    expression_scale: float = 1.0
    enhancer: Optional[str] = None
    still: bool = False
    verbose: bool = False
    cpu: bool = False
    keep_intermediate: bool = False


def _find_conda_executable(explicit_path: Optional[Path]) -> Path:
    candidates = []
    if explicit_path is not None:
        candidates.append(explicit_path.expanduser())

    conda_env_value = os.environ.get("CONDA_EXE")
    if conda_env_value:
        candidates.append(Path(conda_env_value).expanduser())

    which_conda = shutil.which("conda")
    if which_conda:
        candidates.append(Path(which_conda))

    candidates.extend(
        Path(candidate).expanduser()
        for candidate in (
            "~/miniconda3/bin/conda",
            "~/anaconda3/bin/conda",
            "~/miniforge3/bin/conda",
            "~/mambaforge/bin/conda",
        )
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not locate the conda executable. Set CONDA_EXE or pass --sadtalker-conda-exe."
    )


def _build_inference_command(
    config: SadTalkerConfig,
    source_image: Path,
    audio_path: Path,
    result_dir: Path,
) -> list[str]:
    command = [
        str(config.conda_executable),
        "run",
        "-n",
        config.conda_env_name,
        "python",
        "inference.py",
        "--driven_audio",
        str(audio_path),
        "--source_image",
        str(source_image),
        "--checkpoint_dir",
        str(config.checkpoint_dir),
        "--result_dir",
        str(result_dir),
        "--preprocess",
        config.preprocess,
        "--size",
        str(config.size),
        "--pose_style",
        str(config.pose_style),
        "--batch_size",
        str(config.batch_size),
        "--expression_scale",
        str(config.expression_scale),
    ]

    if config.enhancer:
        command.extend(["--enhancer", config.enhancer])
    if config.still:
        command.append("--still")
    if config.verbose:
        command.append("--verbose")
    if config.cpu:
        command.append("--cpu")

    return command


def _resolve_output_video(result_dir: Path) -> Path:
    candidates = sorted(result_dir.glob("*.mp4"), key=lambda item: item.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"SadTalker finished without producing an MP4 in {result_dir}."
        )
    return candidates[-1]


def _run_command(command: Sequence[str], cwd: Path, env: dict[str, str]) -> None:
    try:
        subprocess.run(command, cwd=cwd, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        rendered_command = " ".join(str(part) for part in command)
        raise RuntimeError(f"SadTalker command failed: {rendered_command}") from exc


def generate_talking_head(
    source_image_path: Path | str,
    audio_path: Path | str,
    output_path: Path | str,
    *,
    sadtalker_dir: Optional[Path | str] = None,
    checkpoint_dir: Optional[Path | str] = None,
    conda_executable: Optional[Path | str] = None,
    conda_env_name: str = DEFAULT_SADTALKER_ENV_NAME,
    preprocess: str = "crop",
    size: int = 256,
    pose_style: int = 0,
    batch_size: int = 2,
    expression_scale: float = 1.0,
    enhancer: Optional[str] = None,
    still: bool = False,
    verbose: bool = False,
    cpu: bool = False,
    keep_intermediate: bool = False,
) -> Path:
    source_image = Path(source_image_path).expanduser().resolve()
    driven_audio = Path(audio_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()

    if not source_image.is_file():
        raise FileNotFoundError(f"Source image was not found: {source_image}")
    if not driven_audio.is_file():
        raise FileNotFoundError(f"Driven audio was not found: {driven_audio}")

    resolved_sadtalker_dir = (
        Path(sadtalker_dir).expanduser().resolve()
        if sadtalker_dir is not None
        else _default_sadtalker_dir().resolve()
    )
    if not resolved_sadtalker_dir.is_dir():
        raise FileNotFoundError(f"SadTalker directory was not found: {resolved_sadtalker_dir}")

    resolved_checkpoint_dir = (
        Path(checkpoint_dir).expanduser().resolve()
        if checkpoint_dir is not None
        else _default_checkpoint_dir(resolved_sadtalker_dir).resolve()
    )
    if not resolved_checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"SadTalker checkpoint directory was not found: {resolved_checkpoint_dir}"
        )

    config = SadTalkerConfig(
        sadtalker_dir=resolved_sadtalker_dir,
        checkpoint_dir=resolved_checkpoint_dir,
        conda_executable=_find_conda_executable(
            Path(conda_executable) if conda_executable is not None else None
        ).resolve(),
        conda_env_name=conda_env_name,
        preprocess=preprocess,
        size=size,
        pose_style=pose_style,
        batch_size=batch_size,
        expression_scale=expression_scale,
        enhancer=enhancer,
        still=still,
        verbose=verbose,
        cpu=cpu,
        keep_intermediate=keep_intermediate,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    result_dir = Path(
        tempfile.mkdtemp(prefix=f"{output_file.stem}_sadtalker_", dir=output_file.parent)
    )
    numba_cache_dir = output_file.parent / ".numba-cache"
    matplotlib_config_dir = output_file.parent / ".matplotlib"
    pycache_dir = output_file.parent / ".pycache"
    numba_cache_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
    pycache_dir.mkdir(parents=True, exist_ok=True)

    command = _build_inference_command(
        config=config,
        source_image=source_image,
        audio_path=driven_audio,
        result_dir=result_dir,
    )
    command_env = os.environ.copy()
    command_env["NUMBA_CACHE_DIR"] = str(numba_cache_dir)
    command_env["MPLCONFIGDIR"] = str(matplotlib_config_dir)
    command_env["PYTHONPYCACHEPREFIX"] = str(pycache_dir)

    try:
        _run_command(command, cwd=config.sadtalker_dir, env=command_env)
        raw_output_video = _resolve_output_video(result_dir)
        shutil.copy2(raw_output_video, output_file)
        return output_file
    finally:
        if not config.keep_intermediate:
            shutil.rmtree(result_dir, ignore_errors=True)
