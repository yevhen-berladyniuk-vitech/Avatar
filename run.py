import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from pipeline import generate_speech, generate_talking_head

DEFAULT_OUTPUT_DIR = Path("samples/output")


def _default_output_stem(source_image: Path) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_stem = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in source_image.stem
    ).strip("-")
    if not sanitized_stem:
        sanitized_stem = "avatar"
    return f"{sanitized_stem}_{timestamp}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a speech WAV and SadTalker MP4 from a single input portrait."
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
        help="Path to the source portrait image.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the generated WAV and MP4 should be written.",
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Base filename for the generated assets. Defaults to the image stem plus a timestamp.",
    )
    parser.add_argument(
        "--tts-engine",
        default="auto",
        choices=["auto", "kokoro", "macos_say"],
        help="TTS backend. 'auto' tries Kokoro first and falls back to macOS 'say'.",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help="Optional voice name for the selected TTS backend.",
    )
    parser.add_argument(
        "--sadtalker-dir",
        default=None,
        help="Path to the SadTalker checkout. Defaults to SADTALKER_DIR or ../SadTalker.",
    )
    parser.add_argument(
        "--sadtalker-checkpoint-dir",
        default=None,
        help="Path to SadTalker checkpoints. Defaults to SADTALKER_CHECKPOINT_DIR or <sadtalker-dir>/checkpoints.",
    )
    parser.add_argument(
        "--sadtalker-conda-env",
        default="sadtalker",
        help="Conda environment name used to run SadTalker.",
    )
    parser.add_argument(
        "--sadtalker-conda-exe",
        default=None,
        help="Path to the conda executable. Defaults to CONDA_EXE or common local installs.",
    )
    parser.add_argument(
        "--preprocess",
        default="crop",
        choices=["crop", "extcrop", "resize", "full", "extfull"],
        help="SadTalker preprocess mode.",
    )
    parser.add_argument(
        "--size",
        default=256,
        type=int,
        help="SadTalker face render size. Common values are 256 and 512.",
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
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force SadTalker to run with --cpu.",
    )
    parser.add_argument(
        "--keep-sadtalker-temp",
        action="store_true",
        help="Keep SadTalker's temporary working directory under the output directory.",
    )
    return parser.parse_args()


def _resolve_output_paths(
    source_image: Path,
    output_dir: Path,
    requested_stem: Optional[str],
) -> tuple[Path, Path]:
    stem = requested_stem or _default_output_stem(source_image)
    return output_dir / f"{stem}.wav", output_dir / f"{stem}.mp4"


def main() -> None:
    args = _parse_args()

    source_image = Path(args.input).expanduser().resolve()
    if not source_image.is_file():
        raise SystemExit(f"Source image was not found: {source_image}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path, mp4_path = _resolve_output_paths(source_image, output_dir, args.output_stem)

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
            engine=args.tts_engine,
        )

    print(f"Generating talking-head MP4 at {mp4_path}...")
    generate_talking_head(
        source_image_path=source_image,
        audio_path=wav_path,
        output_path=mp4_path,
        sadtalker_dir=args.sadtalker_dir,
        checkpoint_dir=args.sadtalker_checkpoint_dir,
        conda_executable=args.sadtalker_conda_exe,
        conda_env_name=args.sadtalker_conda_env,
        preprocess=args.preprocess,
        size=args.size,
        enhancer=args.enhancer,
        still=args.still,
        cpu=args.cpu,
        keep_intermediate=args.keep_sadtalker_temp,
    )

    print(f"WAV output: {wav_path}")
    print(f"MP4 output: {mp4_path}")


if __name__ == "__main__":
    main()
