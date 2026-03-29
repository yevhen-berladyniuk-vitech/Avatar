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
DEFAULT_PREPROCESS = "crop"
DEFAULT_SIZE = 256
LEGACY_CHECKPOINT_FILES = (
    "auido2exp_00300-model.pth",
    "auido2pose_00140-model.pth",
    "epoch_20.pth",
    "facevid2vid_00189-model.pth.tar",
    "wav2lip.pth",
)
SADTALKER_REQUIRED_MODULES = ("cv2", "numpy", "safetensors", "torch")


@dataclass(frozen=True)
class SadTalkerConfig:
    sadtalker_dir: Path
    checkpoint_dir: Path
    python_executable: Path
    pose_style: int = 0
    batch_size: int = 2
    size: int = DEFAULT_SIZE
    preprocess: str = DEFAULT_PREPROCESS
    expression_scale: float = 1.0
    enhancer: Optional[str] = None
    background_enhancer: Optional[str] = None
    still: bool = False
    verbose: bool = False
    cpu: bool = False
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


def _candidate_python_paths(sadtalker_dir: Path) -> list[Path]:
    home_dir = Path.home()
    return [
        sadtalker_dir / ".venv" / "bin" / "python",
        sadtalker_dir / "venv" / "bin" / "python",
        sadtalker_dir / "env" / "bin" / "python",
        home_dir / "miniconda3" / "envs" / "sadtalker" / "bin" / "python",
        home_dir / "anaconda3" / "envs" / "sadtalker" / "bin" / "python",
        home_dir / ".conda" / "envs" / "sadtalker" / "bin" / "python",
    ]


def _python_supports_sadtalker(python_executable: Path) -> bool:
    module_check = (
        "import importlib.util, sys; "
        f"mods={SADTALKER_REQUIRED_MODULES!r}; "
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
    sadtalker_dir: Path,
) -> Path:
    candidates: list[Union[Path, str]] = []
    if explicit_path is not None:
        candidates.append(explicit_path)

    python_env_value = os.environ.get("SADTALKER_PYTHON")
    if python_env_value:
        candidates.append(python_env_value)

    candidates.extend(_candidate_python_paths(sadtalker_dir))

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
            if _python_supports_sadtalker(resolved_candidate):
                return resolved_candidate

    if fallback_candidate is not None:
        missing_modules = ", ".join(SADTALKER_REQUIRED_MODULES)
        raise RuntimeError(
            "Could not find a Python interpreter with the modules SadTalker needs "
            f"({missing_modules}). Pass --sadtalker-python or set SADTALKER_PYTHON "
            "to the SadTalker environment."
        )

    raise FileNotFoundError(
        "Could not locate a Python executable for SadTalker. "
        "Pass --sadtalker-python or set SADTALKER_PYTHON."
    )


def _find_sadtalker_dir(explicit_path: Optional[Union[Path, str]]) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path).expanduser())

    sadtalker_env_value = os.environ.get("SADTALKER_DIR")
    if sadtalker_env_value:
        candidates.append(Path(sadtalker_env_value).expanduser())

    candidates.append(AVATAR_ROOT.parent / "SadTalker")

    for candidate in candidates:
        resolved_candidate = candidate.resolve()
        if (resolved_candidate / "inference.py").is_file():
            return resolved_candidate

    raise FileNotFoundError(
        "Could not locate the SadTalker checkout. "
        "Pass --sadtalker-dir or set SADTALKER_DIR."
    )


def _find_checkpoint_dir(
    explicit_path: Optional[Union[Path, str]],
    sadtalker_dir: Path,
) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path).expanduser())

    checkpoint_env_value = os.environ.get("SADTALKER_CHECKPOINT_DIR")
    if checkpoint_env_value:
        candidates.append(Path(checkpoint_env_value).expanduser())

    candidates.append(sadtalker_dir / "checkpoints")

    for candidate in candidates:
        resolved_candidate = candidate.resolve()
        if resolved_candidate.is_dir():
            return resolved_candidate

    expected_default = (sadtalker_dir / "checkpoints").resolve()
    raise FileNotFoundError(
        "Could not locate the SadTalker checkpoints directory. "
        f"Pass --checkpoint-dir, set SADTALKER_CHECKPOINT_DIR, or populate {expected_default}."
    )


def _validate_checkpoint_dir(checkpoint_dir: Path, preprocess: str, size: int) -> None:
    mapping_checkpoint_name = (
        "mapping_00109-model.pth.tar" if "full" in preprocess.lower() else "mapping_00229-model.pth.tar"
    )
    mapping_checkpoint = checkpoint_dir / mapping_checkpoint_name
    if not mapping_checkpoint.is_file():
        raise FileNotFoundError(
            f"SadTalker mapping checkpoint was not found: {mapping_checkpoint}"
        )

    requested_safetensor = checkpoint_dir / f"SadTalker_V0.0.2_{size}.safetensors"
    if requested_safetensor.is_file():
        return

    missing_legacy_files = [
        filename
        for filename in LEGACY_CHECKPOINT_FILES
        if not (checkpoint_dir / filename).is_file()
    ]
    if missing_legacy_files:
        missing_files = ", ".join(missing_legacy_files)
        raise FileNotFoundError(
            "SadTalker checkpoints are incomplete. "
            f"Expected {requested_safetensor.name} or the legacy files in {checkpoint_dir}. "
            f"Missing: {missing_files}"
        )


def _build_inference_command(
    config: SadTalkerConfig,
    source_media: Path,
    audio_path: Path,
    result_dir: Path,
) -> list[str]:
    inference_script = config.sadtalker_dir / "inference.py"
    command = [
        str(config.python_executable),
        "-c",
        _build_inference_bootstrap(enable_enhancer=bool(config.enhancer)),
        str(inference_script),
        "--driven_audio",
        str(audio_path),
        "--source_image",
        str(source_media),
        "--checkpoint_dir",
        str(config.checkpoint_dir),
        "--result_dir",
        str(result_dir),
        "--pose_style",
        str(config.pose_style),
        "--batch_size",
        str(config.batch_size),
        "--size",
        str(config.size),
        "--preprocess",
        config.preprocess,
        "--expression_scale",
        str(config.expression_scale),
    ]

    if config.enhancer:
        command.extend(["--enhancer", config.enhancer])
    if config.background_enhancer:
        command.extend(["--background_enhancer", config.background_enhancer])
    if config.still:
        command.append("--still")
    if config.verbose:
        command.append("--verbose")
    if config.cpu:
        command.append("--cpu")

    return command


def _build_inference_bootstrap(*, enable_enhancer: bool) -> str:
    bootstrap_lines = [
        "import importlib",
        "import importlib.util",
        "import numpy as np",
        "import runpy",
        "import sys",
        "import types",
        "numpy_aliases = {",
        "    'bool': bool,",
        "    'complex': complex,",
        "    'float': float,",
        "    'int': int,",
        "    'object': object,",
        "    'str': str,",
        "}",
        "for alias_name, alias_value in numpy_aliases.items():",
        "    if alias_name not in np.__dict__:",
        "        setattr(np, alias_name, alias_value)",
        "from src.face3d.util import preprocess as face3d_preprocess",
        "_original_pos = face3d_preprocess.POS",
        "def _avatar_pos(xp, x):",
        "    t, s = _original_pos(xp, x)",
        "    t = np.asarray(t, dtype=np.float32).reshape(-1)",
        "    s = np.asarray(s, dtype=np.float32).reshape(())",
        "    return t, s",
        "face3d_preprocess.POS = _avatar_pos",
    ]

    if enable_enhancer:
        bootstrap_lines.extend(
            [
                "if importlib.util.find_spec('torchvision.transforms.functional_tensor') is None:",
                "    try:",
                "        functional_tensor = importlib.import_module('torchvision.transforms._functional_tensor')",
                "    except ModuleNotFoundError:",
                "        functional_tensor = importlib.import_module('torchvision.transforms.functional')",
                "    sys.modules['torchvision.transforms.functional_tensor'] = functional_tensor",
            ]
        )
    else:
        bootstrap_lines.extend(
            [
                "gfpgan_module = types.ModuleType('gfpgan')",
                "class GFPGANer:",
                "    def __init__(self, *args, **kwargs):",
                "        raise RuntimeError('SadTalker enhancer support is unavailable in this runtime. Re-run with a compatible GFPGAN/torchvision stack if you need --enhancer.')",
                "gfpgan_module.GFPGANer = GFPGANer",
                "sys.modules.setdefault('gfpgan', gfpgan_module)",
            ]
        )

    bootstrap_lines.extend(
        [
            "inference_path = sys.argv[1]",
            "sys.argv = sys.argv[1:]",
            "runpy.run_path(inference_path, run_name='__main__')",
        ]
    )
    return "\n".join(bootstrap_lines)


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


def _build_environment(config: SadTalkerConfig) -> dict[str, str]:
    env = os.environ.copy()
    env["SADTALKER_DIR"] = str(config.sadtalker_dir)
    env["SADTALKER_CHECKPOINT_DIR"] = str(config.checkpoint_dir)
    env["PATH"] = os.pathsep.join(
        [
            str(config.python_executable.parent),
            env.get("PATH", ""),
        ]
    )

    existing_pythonpath = env.get("PYTHONPATH")
    pythonpath_entries = [str(config.sadtalker_dir)]
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    return env


def _run_command(command: Sequence[str], cwd: Path, env: dict[str, str]) -> None:
    try:
        subprocess.run(command, cwd=cwd, check=True, env=env)
    except FileNotFoundError as exc:
        rendered_command = " ".join(str(part) for part in command)
        raise RuntimeError(f"Failed to launch SadTalker command: {rendered_command}") from exc
    except subprocess.CalledProcessError as exc:
        rendered_command = " ".join(str(part) for part in command)
        raise RuntimeError(f"SadTalker command failed: {rendered_command}") from exc


def generate_talking_head(
    source_media_path: Union[Path, str],
    audio_path: Union[Path, str],
    output_path: Union[Path, str],
    *,
    sadtalker_dir: Optional[Union[Path, str]] = None,
    checkpoint_dir: Optional[Union[Path, str]] = None,
    sadtalker_python: Optional[Union[Path, str]] = None,
    pose_style: int = 0,
    batch_size: int = 2,
    size: int = DEFAULT_SIZE,
    preprocess: str = DEFAULT_PREPROCESS,
    expression_scale: float = 1.0,
    enhancer: Optional[str] = None,
    background_enhancer: Optional[str] = None,
    still: bool = False,
    verbose: bool = False,
    cpu: bool = False,
    keep_intermediate: bool = False,
) -> Path:
    source_media = Path(source_media_path).expanduser().resolve()
    driven_audio = Path(audio_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()

    if not source_media.is_file():
        raise FileNotFoundError(f"Source input was not found: {source_media}")
    if not driven_audio.is_file():
        raise FileNotFoundError(f"Driven audio was not found: {driven_audio}")

    resolved_sadtalker_dir = _find_sadtalker_dir(sadtalker_dir)
    resolved_checkpoint_dir = _find_checkpoint_dir(checkpoint_dir, resolved_sadtalker_dir)
    _validate_checkpoint_dir(resolved_checkpoint_dir, preprocess, size)

    config = SadTalkerConfig(
        sadtalker_dir=resolved_sadtalker_dir,
        checkpoint_dir=resolved_checkpoint_dir,
        python_executable=_find_python_executable(sadtalker_python, resolved_sadtalker_dir),
        pose_style=pose_style,
        batch_size=batch_size,
        size=size,
        preprocess=preprocess,
        expression_scale=expression_scale,
        enhancer=enhancer,
        background_enhancer=background_enhancer,
        still=still,
        verbose=verbose,
        cpu=cpu,
        keep_intermediate=keep_intermediate,
    )
    print(f"Using SadTalker Python: {config.python_executable}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    workspace_dir = Path(
        tempfile.mkdtemp(prefix=f"{output_file.stem}_sadtalker_", dir=output_file.parent)
    )
    staged_source_media = workspace_dir / f"source{source_media.suffix.lower()}"
    staged_audio_path = workspace_dir / f"driven_audio{driven_audio.suffix.lower()}"
    result_dir = workspace_dir / "results"
    shutil.copy2(source_media, staged_source_media)
    shutil.copy2(driven_audio, staged_audio_path)
    result_dir.mkdir(parents=True, exist_ok=True)

    command = _build_inference_command(
        config=config,
        source_media=staged_source_media,
        audio_path=staged_audio_path,
        result_dir=result_dir,
    )

    try:
        _run_command(command, cwd=config.sadtalker_dir, env=_build_environment(config))
        raw_output_video = _resolve_output_video(result_dir)
        shutil.copy2(raw_output_video, output_file)
        return output_file
    finally:
        if not config.keep_intermediate:
            shutil.rmtree(workspace_dir, ignore_errors=True)
