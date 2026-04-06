from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from pipeline import (
    generate_background_video,
    generate_hologram_video,
    generate_lip_sync,
    generate_speech,
    generate_talking_head,
)

DEFAULT_OUTPUT_DIR = Path("samples/output")
START_VIDEO_STAGES = ("sadtalker", "wav2lip", "background")
PIPELINE_STAGES = ("sadtalker", "wav2lip", "background", "style")
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Avatar pipeline through SadTalker, Wav2Lip refinement, "
            "background matting, and the hologram styling pass."
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
        "--sadtalker-dir",
        default=None,
        help="Path to the local SadTalker checkout. Defaults to SADTALKER_DIR.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Path to the SadTalker checkpoints directory. Defaults to SADTALKER_CHECKPOINT_DIR or <sadtalker-dir>/checkpoints.",
    )
    parser.add_argument(
        "--sadtalker-python",
        default=None,
        help="Python executable used to run SadTalker locally. Defaults to SADTALKER_PYTHON or the current Python interpreter.",
    )
    parser.add_argument(
        "--start-stage",
        default="sadtalker",
        choices=START_VIDEO_STAGES,
        help=(
            "First video stage to run. Use 'wav2lip' to skip SadTalker, or "
            "'background' to skip both SadTalker and Wav2Lip."
        ),
    )
    parser.add_argument(
        "--end-stage",
        default="style",
        choices=PIPELINE_STAGES,
        help=(
            "Last video stage to run. Use 'wav2lip' to stop before background "
            "removal and styling, or 'background' to stop before styling."
        ),
    )
    parser.add_argument(
        "--expression-scale",
        default=1.0,
        type=float,
        help="SadTalker expression scale.",
    )
    parser.add_argument(
        "--preprocess",
        default="extcrop",
        choices=["crop", "extcrop", "resize", "full", "extfull"],
        help="SadTalker preprocessing mode for the source image or first video frame.",
    )
    parser.add_argument(
        "--size",
        default=256,
        type=int,
        choices=[256, 512],
        help="SadTalker face model resolution.",
    )
    parser.add_argument(
        "--enhancer",
        default=None,
        choices=["gfpgan", "RestoreFormer"],
        help="Optional SadTalker face enhancer.",
    )
    parser.add_argument(
        "--still",
        action="store_true",
        help="Use SadTalker still mode for less aggressive motion.",
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


def _validate_stage_range(start_stage: str, end_stage: str) -> None:
    start_index = PIPELINE_STAGES.index(start_stage)
    end_index = PIPELINE_STAGES.index(end_stage)
    if end_index < start_index:
        raise SystemExit(
            f"Invalid stage range: start stage '{start_stage}' comes after end stage "
            f"'{end_stage}'."
        )


def _requested_video_stages(start_stage: str, end_stage: str) -> tuple[str, ...]:
    start_index = PIPELINE_STAGES.index(start_stage)
    end_index = PIPELINE_STAGES.index(end_stage)
    return PIPELINE_STAGES[start_index : end_index + 1]


def _stage_output_path(
    final_output_path: Path,
    stage_name: str,
    requested_stages: tuple[str, ...],
    intermediate_outputs: list[Path],
) -> Path:
    if requested_stages[-1] == stage_name:
        return final_output_path

    stage_output_path = _create_intermediate_output_path(final_output_path, stage_name)
    intermediate_outputs.append(stage_output_path)
    return stage_output_path


def main() -> None:
    args = _parse_args()

    source_media = Path(args.input).expanduser().resolve()
    _validate_source_media(source_media)
    _validate_start_stage_input(source_media, args.start_stage)
    _validate_stage_range(args.start_stage, args.end_stage)

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
    requested_stages = _requested_video_stages(args.start_stage, args.end_stage)

    try:
        if "sadtalker" in requested_stages:
            sadtalker_output_path = _stage_output_path(
                mp4_path,
                "sadtalker",
                requested_stages,
                intermediate_outputs,
            )

            print(f"Generating talking-head MP4 at {sadtalker_output_path}...")
            generate_talking_head(
                source_media_path=source_media,
                audio_path=wav_path,
                output_path=sadtalker_output_path,
                sadtalker_dir=args.sadtalker_dir,
                checkpoint_dir=args.checkpoint_dir,
                sadtalker_python=args.sadtalker_python,
                expression_scale=args.expression_scale,
                preprocess=args.preprocess,
                size=args.size,
                enhancer=args.enhancer,
                still=args.still,
                cpu=False,
                keep_intermediate=False,
            )
            active_video_input = sadtalker_output_path

        if "wav2lip" in requested_stages:
            wav2lip_output_path = _stage_output_path(
                mp4_path,
                "wav2lip",
                requested_stages,
                intermediate_outputs,
            )

            print(f"Generating lip-synced MP4 at {wav2lip_output_path}...")
            generate_lip_sync(
                source_media_path=active_video_input,
                audio_path=wav_path,
                output_path=wav2lip_output_path,
            )
            active_video_input = wav2lip_output_path

        if "background" in requested_stages:
            background_output_path = _stage_output_path(
                mp4_path,
                "background",
                requested_stages,
                intermediate_outputs,
            )

            print(f"Generating background-matted MP4 at {background_output_path}...")
            generate_background_video(
                source_media_path=active_video_input,
                output_path=background_output_path,
            )
            active_video_input = background_output_path

        if "style" in requested_stages:
            print(f"Generating hologram-styled MP4 at {mp4_path}...")
            generate_hologram_video(
                source_media_path=active_video_input,
                output_path=mp4_path,
            )
    finally:
        for intermediate_output in intermediate_outputs:
            intermediate_output.unlink(missing_ok=True)

    print(f"WAV output: {wav_path}")
    print(f"MP4 output: {mp4_path}")


if __name__ == "__main__":
    main()
