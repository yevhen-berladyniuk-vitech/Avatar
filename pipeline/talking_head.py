from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

AVATAR_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SADTALKER_DOCKER_IMAGE = "wawa9000/sadtalker"
CONTAINER_HOST_DIR = Path("/host_dir")
CONTAINER_RESULT_DIR = CONTAINER_HOST_DIR / "results"
CONTAINER_PATCH_ENTRY = Path("/app/SadTalker/run_with_avatar_patch.py")
LOCAL_PATCH_ENTRY = AVATAR_ROOT / "pipeline" / "sadtalker_container_entry.py"


def _default_docker_image() -> str:
    return os.environ.get("SADTALKER_DOCKER_IMAGE", DEFAULT_SADTALKER_DOCKER_IMAGE)


@dataclass(frozen=True)
class SadTalkerConfig:
    docker_executable: Path
    docker_image: str = DEFAULT_SADTALKER_DOCKER_IMAGE
    docker_gpus: Optional[str] = None
    docker_platform: Optional[str] = None
    pose_style: int = 0
    batch_size: int = 2
    expression_scale: float = 1.0
    enhancer: Optional[str] = None
    still: bool = False
    verbose: bool = False
    cpu: bool = False
    keep_intermediate: bool = False


def _resolve_executable(candidate: Path | str) -> Optional[Path]:
    candidate_text = str(candidate)
    direct_path = Path(candidate_text).expanduser()
    if direct_path.is_file():
        return direct_path.resolve()

    discovered_path = shutil.which(candidate_text)
    if discovered_path:
        return Path(discovered_path).resolve()

    return None


def _find_docker_executable(explicit_path: Optional[Path | str]) -> Path:
    candidates = []
    if explicit_path is not None:
        candidates.append(explicit_path)

    docker_env_value = os.environ.get("DOCKER_EXE")
    if docker_env_value:
        candidates.append(docker_env_value)

    candidates.append("docker")
    candidates.extend(
        [
            "/Applications/Docker.app/Contents/Resources/bin/docker",
            "/usr/local/bin/docker",
            "/opt/homebrew/bin/docker",
        ]
    )

    for candidate in candidates:
        resolved_candidate = _resolve_executable(candidate)
        if resolved_candidate is not None:
            return resolved_candidate

    raise FileNotFoundError(
        "Could not locate the docker executable. Install Docker or pass --docker-executable."
    )


def _containerize_path(host_root: Path, host_path: Path) -> str:
    return str(CONTAINER_HOST_DIR / host_path.relative_to(host_root))


def _default_docker_platform() -> Optional[str]:
    configured_platform = os.environ.get("SADTALKER_DOCKER_PLATFORM")
    if configured_platform:
        return configured_platform

    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return "linux/amd64"

    return None


def _build_inference_command(
    config: SadTalkerConfig,
    mounted_host_dir: Path,
    patch_entry_path: Path,
    source_image: Path,
    audio_path: Path,
    result_dir: Path,
) -> list[str]:
    command = [
        str(config.docker_executable),
        "run",
        "--rm",
    ]

    if config.docker_gpus:
        command.extend(["--gpus", config.docker_gpus])
    if config.docker_platform:
        command.extend(["--platform", config.docker_platform])

    command.extend(
        [
            "-v",
            f"{mounted_host_dir}:{CONTAINER_HOST_DIR}",
            "-v",
            f"{patch_entry_path}:{CONTAINER_PATCH_ENTRY}:ro",
            "--entrypoint",
            "python3",
            config.docker_image,
            str(CONTAINER_PATCH_ENTRY),
        ]
    )

    command.extend(
        [
            "--driven_audio",
            _containerize_path(mounted_host_dir, audio_path),
            "--source_image",
            _containerize_path(mounted_host_dir, source_image),
            "--result_dir",
            _containerize_path(mounted_host_dir, result_dir),
            "--pose_style",
            str(config.pose_style),
            "--batch_size",
            str(config.batch_size),
            "--expression_scale",
            str(config.expression_scale),
        ]
    )

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
    candidates = sorted(result_dir.rglob("*.mp4"), key=lambda item: item.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"SadTalker finished without producing an MP4 in {result_dir}."
        )
    final_candidates = [candidate for candidate in candidates if "temp_" not in candidate.name]
    if final_candidates:
        return final_candidates[-1]
    return candidates[-1]


def _run_command(command: Sequence[str], cwd: Path, env: dict[str, str]) -> None:
    try:
        subprocess.run(command, cwd=cwd, check=True, env=env)
    except FileNotFoundError as exc:
        rendered_command = " ".join(str(part) for part in command)
        raise RuntimeError(f"Failed to launch Docker command: {rendered_command}") from exc
    except subprocess.CalledProcessError as exc:
        rendered_command = " ".join(str(part) for part in command)
        raise RuntimeError(f"SadTalker command failed: {rendered_command}") from exc


def generate_talking_head(
    source_image_path: Path | str,
    audio_path: Path | str,
    output_path: Path | str,
    *,
    docker_executable: Optional[Path | str] = None,
    docker_image: Optional[str] = None,
    docker_gpus: Optional[str] = None,
    docker_platform: Optional[str] = None,
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

    config = SadTalkerConfig(
        docker_executable=_find_docker_executable(docker_executable),
        docker_image=docker_image or _default_docker_image(),
        docker_gpus=docker_gpus or os.environ.get("SADTALKER_DOCKER_GPUS") or None,
        docker_platform=docker_platform or _default_docker_platform(),
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
    workspace_dir = Path(
        tempfile.mkdtemp(prefix=f"{output_file.stem}_sadtalker_", dir=output_file.parent)
    )
    if not LOCAL_PATCH_ENTRY.is_file():
        raise FileNotFoundError(f"SadTalker patch entry was not found: {LOCAL_PATCH_ENTRY}")
    staged_source_image = workspace_dir / f"source_image{source_image.suffix}"
    staged_audio_path = workspace_dir / f"driven_audio{driven_audio.suffix}"
    result_dir = workspace_dir / CONTAINER_RESULT_DIR.name
    shutil.copy2(source_image, staged_source_image)
    shutil.copy2(driven_audio, staged_audio_path)
    result_dir.mkdir(parents=True, exist_ok=True)

    command = _build_inference_command(
        config=config,
        mounted_host_dir=workspace_dir,
        patch_entry_path=LOCAL_PATCH_ENTRY.resolve(),
        source_image=staged_source_image,
        audio_path=staged_audio_path,
        result_dir=result_dir,
    )

    try:
        _run_command(command, cwd=workspace_dir, env=os.environ.copy())
        raw_output_video = _resolve_output_video(result_dir)
        shutil.copy2(raw_output_video, output_file)
        return output_file
    finally:
        if not config.keep_intermediate:
            shutil.rmtree(workspace_dir, ignore_errors=True)
