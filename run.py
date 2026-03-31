from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from pipeline import (
    generate_background_video,
    generate_echo_mimic,
    generate_speech,
)

DEFAULT_OUTPUT_DIR = Path("samples/output")
VIDEO_STAGES = ("echomimic", "background")
VIDEO_SOURCE_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4"}
SUPPORTED_SOURCE_EXTENSIONS = {
    ".avi",
    ".bmp",
    ".jpeg",
    ".jpg",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".png",
    ".webp",
}


def _resolve_executable(candidate: str) -> Optional[Path]:
    direct_path = Path(candidate).expanduser()
    if direct_path.is_file():
        return direct_path.resolve()

    discovered_path = shutil.which(candidate)
    if discovered_path:
        return Path(discovered_path).resolve()

    return None


def _find_ffmpeg_executable() -> Path:
    candidates: list[str] = []

    ffmpeg_env_value = os.environ.get("FFMPEG")
    if ffmpeg_env_value:
        candidates.append(ffmpeg_env_value)

    ffmpeg_path_env_value = os.environ.get("FFMPEG_PATH")
    if ffmpeg_path_env_value:
        ffmpeg_path = Path(ffmpeg_path_env_value).expanduser()
        if ffmpeg_path.is_dir():
            candidates.append(str(ffmpeg_path / "ffmpeg"))
        else:
            candidates.append(str(ffmpeg_path))

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(str(Path(conda_prefix) / "bin" / "ffmpeg"))

    home_dir = Path.home()
    candidates.extend(
        [
            str(home_dir / "miniconda3" / "envs" / "echomimic" / "bin" / "ffmpeg"),
            str(home_dir / "anaconda3" / "envs" / "echomimic" / "bin" / "ffmpeg"),
            "ffmpeg",
        ]
    )

    for candidate in candidates:
        resolved_candidate = _resolve_executable(candidate)
        if resolved_candidate is not None:
            return resolved_candidate

    raise FileNotFoundError(
        "Could not locate an ffmpeg executable. Install ffmpeg or set FFMPEG_PATH."
    )


def _default_output_stem(source_media: Path) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_stem = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in source_media.stem
    ).strip("-")
    if not sanitized_stem:
        sanitized_stem = "avatar"
    return f"{sanitized_stem}_{timestamp}"


def _create_intermediate_output_path(output_path: Path, stage_name: str) -> Path:
    temp_handle, temp_path = tempfile.mkstemp(
        prefix=f"{output_path.stem}_{stage_name}_",
        suffix=".mp4",
        dir=output_path.parent,
    )
    os.close(temp_handle)
    intermediate_path = Path(temp_path).resolve()
    intermediate_path.unlink(missing_ok=True)
    return intermediate_path


def _mux_audio_into_video(video_path: Path, audio_path: Path, output_path: Path) -> None:
    ffmpeg_executable = _find_ffmpeg_executable()
    command = [
        str(ffmpeg_executable),
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        rendered_command = " ".join(command)
        raise RuntimeError(
            "Failed to mux audio into the final MP4. "
            f"Command: {rendered_command}. stderr: {(exc.stderr or '').strip()}"
        ) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Avatar pipeline through EchoMimic and background matting."
        )
    )
    speech_source_group = parser.add_mutually_exclusive_group(required=True)
    speech_source_group.add_argument(
        "--text",
        help="Text that the avatar should speak.",
    )
    speech_source_group.add_argument(
        "--audio",
        help="Optional path to a pre-generated WAV file. If provided, TTS is skipped and the WAV is copied to the output path.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the source image or video for the selected start stage.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the generated WAV and MP4 should be written.",
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Base filename for the generated assets. Defaults to the input stem plus a timestamp.",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help="Optional voice name for the selected TTS backend.",
    )
    parser.add_argument(
        "--echomimic-dir",
        default=None,
        help="Path to the local EchoMimic checkout. Defaults to ECHOMIMIC_DIR.",
    )
    parser.add_argument(
        "--echomimic-python",
        default=None,
        help="Python executable used to run EchoMimic locally. Defaults to ECHOMIMIC_PYTHON or the discovered echomimic environment.",
    )
    parser.add_argument(
        "--echomimic-variant",
        default="accelerated",
        choices=["standard", "accelerated"],
        help="Which EchoMimic audio-driven pipeline to use.",
    )
    parser.add_argument(
        "--start-stage",
        default="echomimic",
        choices=VIDEO_STAGES,
        help=(
            "First video stage to run. Use 'background' to skip EchoMimic and "
            "start from an existing video."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device string passed to EchoMimic, for example 'cpu', 'mps', 'auto', or 'cuda:0'.",
    )
    parser.add_argument(
        "--width",
        default=512,
        type=int,
        help="EchoMimic output width.",
    )
    parser.add_argument(
        "--height",
        default=512,
        type=int,
        help="EchoMimic output height.",
    )
    parser.add_argument(
        "--length",
        default=1200,
        type=int,
        help="EchoMimic maximum generation length in frames.",
    )
    parser.add_argument(
        "--seed",
        default=420,
        type=int,
        help="EchoMimic random seed.",
    )
    parser.add_argument(
        "--cfg",
        default=None,
        type=float,
        help="EchoMimic CFG value. Defaults depend on the selected variant.",
    )
    parser.add_argument(
        "--steps",
        default=None,
        type=int,
        help="EchoMimic denoising steps. Defaults depend on the selected variant.",
    )
    parser.add_argument(
        "--fps",
        default=24,
        type=int,
        help="EchoMimic output FPS.",
    )
    parser.add_argument(
        "--sample-rate",
        default=16000,
        type=int,
        help="Audio sample rate passed to EchoMimic.",
    )
    parser.add_argument(
        "--context-frames",
        default=12,
        type=int,
        help="EchoMimic context window size.",
    )
    parser.add_argument(
        "--context-overlap",
        default=3,
        type=int,
        help="EchoMimic context overlap.",
    )
    parser.add_argument(
        "--facemask-dilation-ratio",
        default=0.1,
        type=float,
        help="EchoMimic face-mask dilation ratio.",
    )
    parser.add_argument(
        "--facecrop-dilation-ratio",
        default=0.5,
        type=float,
        help="EchoMimic face-crop dilation ratio.",
    )
    return parser.parse_args()


def _resolve_output_paths(
    source_media: Path,
    output_dir: Path,
    requested_stem: Optional[str],
) -> tuple[Path, Path]:
    stem = requested_stem or _default_output_stem(source_media)
    return output_dir / f"{stem}.wav", output_dir / f"{stem}.mp4"


def _validate_source_media(source_media: Path) -> None:
    if not source_media.is_file():
        raise SystemExit(f"Source input was not found: {source_media}")

    if source_media.suffix.lower() not in SUPPORTED_SOURCE_EXTENSIONS:
        supported_extensions = ", ".join(sorted(SUPPORTED_SOURCE_EXTENSIONS))
        raise SystemExit(
            f"Unsupported input type for {source_media}. Use one of: {supported_extensions}"
        )


def _validate_start_stage_input(source_media: Path, start_stage: str) -> None:
    if start_stage == "background" and source_media.suffix.lower() not in VIDEO_SOURCE_EXTENSIONS:
        supported_video_extensions = ", ".join(sorted(VIDEO_SOURCE_EXTENSIONS))
        raise SystemExit(
            "The background stage requires a video input. "
            f"Use one of: {supported_video_extensions}"
        )


def _requested_video_stages(start_stage: str) -> tuple[str, ...]:
    start_index = VIDEO_STAGES.index(start_stage)
    return VIDEO_STAGES[start_index:]


def main() -> None:
    args = _parse_args()

    source_media = Path(args.input).expanduser().resolve()
    _validate_source_media(source_media)
    _validate_start_stage_input(source_media, args.start_stage)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path, mp4_path = _resolve_output_paths(source_media, output_dir, args.output_stem)

    if args.audio:
        source_audio = Path(args.audio).expanduser().resolve()
        if not source_audio.is_file():
            raise SystemExit(f"Input audio was not found: {source_audio}")

        print(f"Copying input WAV to {wav_path}...")
        shutil.copy2(source_audio, wav_path)
    else:
        print(f"Generating speech WAV at {wav_path}...")
        generate_speech(
            text=args.text,
            output_path=wav_path,
            voice=args.voice,
        )

    active_video_input = source_media
    intermediate_outputs: list[Path] = []
    requested_stages = _requested_video_stages(args.start_stage)

    try:
        if "echomimic" in requested_stages:
            echomimic_output_path = _create_intermediate_output_path(mp4_path, "echomimic")
            intermediate_outputs.append(echomimic_output_path)

            print(f"Generating EchoMimic MP4 at {echomimic_output_path}...")
            generate_echo_mimic(
                source_media_path=source_media,
                audio_path=wav_path,
                output_path=echomimic_output_path,
                echomimic_dir=args.echomimic_dir,
                echomimic_python=args.echomimic_python,
                variant=args.echomimic_variant,
                width=args.width,
                height=args.height,
                length=args.length,
                seed=args.seed,
                facemask_dilation_ratio=args.facemask_dilation_ratio,
                facecrop_dilation_ratio=args.facecrop_dilation_ratio,
                context_frames=args.context_frames,
                context_overlap=args.context_overlap,
                cfg=args.cfg,
                steps=args.steps,
                sample_rate=args.sample_rate,
                fps=args.fps,
                device=args.device,
                keep_intermediate=False,
            )
            active_video_input = echomimic_output_path

        if "background" in requested_stages:
            background_output_path = _create_intermediate_output_path(mp4_path, "background")
            intermediate_outputs.append(background_output_path)

            print(f"Generating background-matted MP4 at {background_output_path}...")
            generate_background_video(
                source_media_path=active_video_input,
                output_path=background_output_path,
            )
            active_video_input = background_output_path

        print(f"Muxing final MP4 with audio at {mp4_path}...")
        _mux_audio_into_video(active_video_input, wav_path, mp4_path)
    finally:
        for intermediate_output in intermediate_outputs:
            intermediate_output.unlink(missing_ok=True)

    print(f"WAV output: {wav_path}")
    print(f"MP4 output: {mp4_path}")


if __name__ == "__main__":
    main()
