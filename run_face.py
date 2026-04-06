from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from pipeline.face_crop import DEFAULT_PADDING_RATIO, crop_face_image
from pipeline.style import (
    DEFAULT_EDGE_STRENGTH,
    DEFAULT_GHOST_STRENGTH,
    DEFAULT_GLOW_STRENGTH,
    DEFAULT_MASK_THRESHOLD,
    DEFAULT_SCANLINE_ALPHA,
    DEFAULT_SCANLINE_SPACING,
    DEFAULT_SUBJECT_OPACITY,
    generate_hologram_video,
)

DEFAULT_OUTPUT_DIR = Path("samples/output")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crop the main face from an image and convert it into a hologram-styled portrait."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the source image.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the hologram image should be written.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=DEFAULT_PADDING_RATIO,
        help="Extra border around the detected face, as a fraction of face size.",
    )
    parser.add_argument(
        "--no-square",
        action="store_true",
        help="Keep the cropped face rectangular instead of forcing a square canvas.",
    )
    parser.add_argument(
        "--mask-threshold",
        default=DEFAULT_MASK_THRESHOLD,
        type=int,
        help="Brightness threshold used by the hologram stage to isolate the subject.",
    )
    parser.add_argument(
        "--glow-strength",
        default=DEFAULT_GLOW_STRENGTH,
        type=float,
        help="Strength of the bloom and aura around the face.",
    )
    parser.add_argument(
        "--edge-strength",
        default=DEFAULT_EDGE_STRENGTH,
        type=float,
        help="Strength of the bright hologram edge outlines.",
    )
    parser.add_argument(
        "--ghost-strength",
        default=DEFAULT_GHOST_STRENGTH,
        type=float,
        help="Strength of the subtle double-image ghosting effect.",
    )
    parser.add_argument(
        "--scanline-alpha",
        default=DEFAULT_SCANLINE_ALPHA,
        type=float,
        help="Darkening amount for the scanline overlay.",
    )
    parser.add_argument(
        "--scanline-spacing",
        default=DEFAULT_SCANLINE_SPACING,
        type=int,
        help="Vertical distance between scanlines in pixels.",
    )
    parser.add_argument(
        "--subject-opacity",
        default=DEFAULT_SUBJECT_OPACITY,
        type=float,
        help="Overall opacity of the styled face before glow is added.",
    )
    return parser.parse_args()


def _resolve_output_path(source_image: Path, output_dir: Path) -> Path:
    return output_dir / f"{source_image.stem}_face_hologram{source_image.suffix.lower()}"


def main() -> None:
    args = _parse_args()

    source_image = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    final_output_path = _resolve_output_path(source_image, output_dir)

    with tempfile.TemporaryDirectory(
        prefix=f"{source_image.stem}_face_",
        dir=output_dir,
    ) as temp_dir:
        cropped_face_path = Path(temp_dir) / f"{source_image.stem}_face.png"

        print(f"Cropping face to {cropped_face_path}...")
        crop_face_image(
            source_image_path=source_image,
            output_path=cropped_face_path,
            padding_ratio=args.padding,
            square=not args.no_square,
            transparent_background=True,
        )

        print(f"Generating hologram image at {final_output_path}...")
        generate_hologram_video(
            source_media_path=cropped_face_path,
            output_path=final_output_path,
            mask_threshold=args.mask_threshold,
            glow_strength=args.glow_strength,
            edge_strength=args.edge_strength,
            ghost_strength=args.ghost_strength,
            scanline_alpha=args.scanline_alpha,
            scanline_spacing=args.scanline_spacing,
            subject_opacity=args.subject_opacity,
        )

    print(f"Hologram output: {final_output_path}")


if __name__ == "__main__":
    main()
