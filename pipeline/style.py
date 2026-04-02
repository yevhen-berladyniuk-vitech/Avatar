from __future__ import annotations

import argparse
import heapq
import importlib.util
import subprocess
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Final, Optional, Union

DEFAULT_FRAME_RATE: Final = 25.0
DEFAULT_GLOW_STRENGTH: Final = 0.55
DEFAULT_EDGE_STRENGTH: Final = 0.35
DEFAULT_GHOST_STRENGTH: Final = 0.18
DEFAULT_SCANLINE_ALPHA: Final = 0.18
DEFAULT_SCANLINE_SPACING: Final = 4
DEFAULT_MASK_THRESHOLD: Final = 8
DEFAULT_SUBJECT_OPACITY: Final = 0.90
DEFAULT_RANDOM_SEED: Final = 1337
DEFAULT_TINT_BGR: Final = (255, 230, 140)
DEFAULT_BACKGROUND_BGR: Final = (110, 42, 16)
VIDEO_SOURCE_EXTENSIONS: Final = {".avi", ".m4v", ".mkv", ".mov", ".mp4"}
REQUIRED_MODULE_PACKAGES: Final = {
    "av": "av",
    "cv2": "opencv-python-headless",
    "numpy": "numpy",
}


def _validate_source_video(source_video: Path) -> None:
    if not source_video.is_file():
        raise FileNotFoundError(f"Style-stage source video was not found: {source_video}")

    if source_video.suffix.lower() not in VIDEO_SOURCE_EXTENSIONS:
        supported_extensions = ", ".join(sorted(VIDEO_SOURCE_EXTENSIONS))
        raise ValueError(
            "The hologram style stage requires a video input. "
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
            "The hologram style stage requires additional Python packages in the "
            f"current environment. Install: {rendered_packages}"
        )


def _resolve_frame_rate(capture: object, cv2_module: object) -> float:
    fps = float(capture.get(cv2_module.CAP_PROP_FPS))
    if fps > 0:
        return fps
    return DEFAULT_FRAME_RATE


def _resolve_writer_fourcc(cv2_module: object, output_file: Path) -> int:
    if output_file.suffix.lower() == ".avi":
        return cv2_module.VideoWriter_fourcc(*"XVID")
    return cv2_module.VideoWriter_fourcc(*"mp4v")


def _packet_timestamp_seconds(packet: object) -> float:
    packet_position = packet.pts if packet.pts is not None else packet.dts
    if packet_position is None or packet.time_base is None:
        return 0.0
    return float(packet_position * packet.time_base)


def _iter_muxable_packets(
    container: object,
    streams: list[object],
    output_stream_by_index: dict[int, object],
):
    for packet in container.demux(streams):
        if packet.dts is None:
            continue
        yield (
            _packet_timestamp_seconds(packet),
            packet,
            output_stream_by_index[packet.stream.index],
        )


def _copy_streams_with_audio(
    styled_video_path: Path,
    source_video_path: Path,
    output_path: Path,
) -> None:
    import av

    with av.open(str(source_video_path)) as source_container:
        audio_streams = [stream for stream in source_container.streams if stream.type == "audio"]
        if not audio_streams:
            shutil.copy2(styled_video_path, output_path)
            return

    with av.open(str(styled_video_path)) as styled_container:
        styled_video_stream = next(
            (stream for stream in styled_container.streams if stream.type == "video"),
            None,
        )
        if styled_video_stream is None:
            raise RuntimeError(f"No styled video stream found in {styled_video_path}")

        with av.open(str(source_video_path)) as source_container:
            audio_streams = [
                stream for stream in source_container.streams if stream.type == "audio"
            ]
            with av.open(str(output_path), mode="w") as output_container:
                output_video_stream = output_container.add_stream_from_template(
                    styled_video_stream
                )
                output_audio_streams = {
                    stream.index: output_container.add_stream_from_template(stream)
                    for stream in audio_streams
                }

                merged_packets = heapq.merge(
                    _iter_muxable_packets(
                        styled_container,
                        [styled_video_stream],
                        {styled_video_stream.index: output_video_stream},
                    ),
                    _iter_muxable_packets(
                        source_container,
                        audio_streams,
                        output_audio_streams,
                    ),
                    key=lambda item: item[0],
                )

                for _, packet, output_stream in merged_packets:
                    packet.stream = output_stream
                    output_container.mux(packet)


def _run_audio_remux_subprocess(
    styled_video_path: Path,
    source_video_path: Path,
    output_path: Path,
) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "__remux_audio__",
        str(styled_video_path),
        str(source_video_path),
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        rendered_command = " ".join(command)
        raise RuntimeError(f"Failed to launch hologram audio remux command: {rendered_command}") from exc
    except subprocess.CalledProcessError as exc:
        rendered_command = " ".join(command)
        raise RuntimeError(f"Hologram audio remux command failed: {rendered_command}") from exc


def _run_internal_remux_command(arguments: list[str]) -> int:
    if len(arguments) != 3:
        raise SystemExit(
            "Internal hologram audio remux mode expects: <styled_video> <source_video> <output>"
        )

    styled_video_path = Path(arguments[0]).expanduser().resolve()
    source_video_path = Path(arguments[1]).expanduser().resolve()
    output_path = Path(arguments[2]).expanduser().resolve()
    _copy_streams_with_audio(
        styled_video_path=styled_video_path,
        source_video_path=source_video_path,
        output_path=output_path,
    )
    return 0


def _largest_connected_component(cv2_module: object, np_module: object, mask: object) -> object:
    label_count, labels, stats, _ = cv2_module.connectedComponentsWithStats(mask, 8)
    if label_count <= 1:
        return mask

    largest_label = 1 + int(np_module.argmax(stats[1:, cv2_module.CC_STAT_AREA]))
    component_mask = np_module.zeros_like(mask)
    component_mask[labels == largest_label] = 255
    return component_mask


def _extract_subject_mask(
    cv2_module: object,
    np_module: object,
    frame: object,
    previous_mask: Optional[object],
    *,
    mask_threshold: int,
) -> object:
    max_channel = frame.max(axis=2)
    binary_mask = np_module.where(max_channel > mask_threshold, 255, 0).astype(np_module.uint8)

    if int(binary_mask.max()) == 0:
        if previous_mask is None:
            return np_module.zeros(frame.shape[:2], dtype=np_module.float32)
        return previous_mask.copy()

    binary_mask = _largest_connected_component(cv2_module, np_module, binary_mask)
    binary_mask = cv2_module.morphologyEx(
        binary_mask,
        cv2_module.MORPH_OPEN,
        np_module.ones((3, 3), dtype=np_module.uint8),
    )
    binary_mask = cv2_module.dilate(
        binary_mask,
        np_module.ones((5, 5), dtype=np_module.uint8),
        iterations=1,
    )

    soft_mask = cv2_module.GaussianBlur(
        binary_mask.astype(np_module.float32) / 255.0,
        (0, 0),
        3.0,
    )
    energy_mask = np_module.clip(
        (
            max_channel.astype(np_module.float32) / 255.0
            - (mask_threshold / 255.0)
        )
        / 0.18,
        0.0,
        1.0,
    )
    soft_mask = np_module.maximum(soft_mask, energy_mask)

    if previous_mask is not None:
        soft_mask = (0.72 * soft_mask) + (0.28 * previous_mask)

    return np_module.clip(soft_mask, 0.0, 1.0)


def _build_scanlines(
    np_module: object,
    *,
    height: int,
    width: int,
    frame_index: int,
    spacing: int,
    alpha: float,
) -> object:
    spacing = max(2, int(spacing))
    rows = np_module.arange(height, dtype=np_module.float32).reshape(height, 1)
    animated_rows = rows + (frame_index * 0.65)
    band_mask = ((animated_rows % spacing) < 1.2).astype(np_module.float32)
    wave = 0.5 + 0.5 * np_module.sin(animated_rows * 0.18)
    scanlines = 1.0 - (alpha * band_mask) - (alpha * 0.10 * wave)
    scanlines = np_module.clip(scanlines, 0.0, 1.0)
    return np_module.repeat(scanlines, width, axis=1)


def _build_digital_rain(
    cv2_module: object,
    np_module: object,
    *,
    height: int,
    width: int,
    frame_index: int,
    mask: object,
) -> object:
    rows = np_module.arange(height, dtype=np_module.float32).reshape(height, 1)
    cols = np_module.arange(width, dtype=np_module.float32).reshape(1, width)
    trail_heads = np_module.mod((frame_index * 7.5) + (cols * 3.4), height)
    trail_lengths = 22.0 + (
        18.0 * (0.5 + 0.5 * np_module.sin((cols * 0.08) + (frame_index * 0.04)))
    )
    trail_body = np_module.clip(
        1.0 - np_module.maximum(rows - trail_heads, 0.0) / trail_lengths,
        0.0,
        1.0,
    )
    trail_body *= (rows >= trail_heads).astype(np_module.float32)
    digit_pattern = (
        np_module.mod((rows * 0.95) + (cols * 0.21) + (frame_index * 2.1), 9.0) < 2.3
    ).astype(np_module.float32)
    droplet_heads = np_module.exp(-((rows - trail_heads) ** 2) / (2.0 * (2.4 ** 2)))
    column_strength = 0.25 + (0.75 * (0.5 + 0.5 * np_module.sin(cols * 0.15)))
    rain = column_strength * ((0.55 * trail_body * digit_pattern) + (0.95 * droplet_heads))

    top_emphasis = np_module.clip(1.15 - (rows / max(height, 1)), 0.35, 1.15)
    subject_field = np_module.clip(
        (mask * 1.2) + (cv2_module.GaussianBlur(mask, (0, 0), 14.0) * 0.35),
        0.0,
        1.0,
    )
    rain *= top_emphasis * subject_field
    return cv2_module.GaussianBlur(rain, (0, 0), 0.8)


def _build_point_cloud(
    cv2_module: object,
    np_module: object,
    contrast_luma: object,
    mask: object,
    *,
    frame_index: int,
) -> object:
    height, width = mask.shape
    step = 6
    points = np_module.zeros((height, width), dtype=np_module.float32)
    sampled_mask = mask[step // 2 :: step, step // 2 :: step]
    sampled_luma = contrast_luma[step // 2 :: step, step // 2 :: step]
    sampled_rows = np_module.arange(sampled_mask.shape[0], dtype=np_module.float32).reshape(-1, 1)
    sampled_cols = np_module.arange(sampled_mask.shape[1], dtype=np_module.float32).reshape(1, -1)
    threshold = 0.16 + (
        0.12
        * (0.5 + 0.5 * np_module.sin((sampled_cols * 0.55) + (sampled_rows * 0.28) + (frame_index * 0.12)))
    )
    sampled_points = (
        (sampled_mask > 0.20) & (sampled_luma > threshold)
    ).astype(np_module.float32)
    points[step // 2 :: step, step // 2 :: step] = sampled_points

    point_glow = cv2_module.GaussianBlur(points, (0, 0), 1.1)
    point_trails = cv2_module.GaussianBlur(points, (0, 0), 0.45, 5.8)
    return np_module.clip((point_glow * 1.65) + (point_trails * 0.90), 0.0, 1.0)


def _build_glitch_overlay(
    cv2_module: object,
    np_module: object,
    subject: object,
    mask: object,
    *,
    frame_index: int,
) -> object:
    height, width = mask.shape
    pixelated = cv2_module.resize(
        subject,
        (max(1, width // 30), max(1, height // 30)),
        interpolation=cv2_module.INTER_LINEAR,
    )
    pixelated = cv2_module.resize(
        pixelated,
        (width, height),
        interpolation=cv2_module.INTER_NEAREST,
    )

    rows = np_module.arange(height, dtype=np_module.float32).reshape(height, 1)
    cols = np_module.arange(width, dtype=np_module.float32).reshape(1, width)
    band_mask = (np_module.mod(rows + (frame_index * 4.8), 57.0) < 5.0).astype(np_module.float32)
    cell_mask = (
        np_module.mod((cols * 0.18) + (rows * 0.11) + (frame_index * 0.9), 7.0) < 1.2
    ).astype(np_module.float32)
    glitch_mask = band_mask * cell_mask * mask
    shifted = np_module.roll(pixelated, shift=int(round(5.0 * np_module.sin(frame_index * 0.22))), axis=1)
    return shifted * glitch_mask[..., None] * 0.28


def _build_dot_matrix_background(
    cv2_module: object,
    np_module: object,
    *,
    height: int,
    width: int,
    frame_index: int,
) -> object:
    step_x = 10
    step_y = 10
    grid_height = max(1, height // step_y + 2)
    grid_width = max(1, width // step_x + 2)

    row_ids = np_module.arange(grid_height, dtype=np_module.float32).reshape(-1, 1)
    col_ids = np_module.arange(grid_width, dtype=np_module.float32).reshape(1, -1)

    head_positions = np_module.mod((frame_index * 0.72) + (col_ids * 1.8), grid_height)
    trail_lengths = 5.0 + (
        6.0 * (0.5 + 0.5 * np_module.sin((col_ids * 0.31) + (frame_index * 0.06)))
    )
    distances = np_module.mod(row_ids - head_positions, grid_height)
    trails = np_module.clip(1.0 - (distances / trail_lengths), 0.0, 1.0)

    row_band = 0.35 + (
        0.65 * (0.5 + 0.5 * np_module.sin((row_ids * 0.42) + (frame_index * 0.04)))
    )
    column_band = 0.25 + (
        0.75 * (0.5 + 0.5 * np_module.sin((col_ids * 0.51) + 0.9))
    )
    sparkles = (
        np_module.mod((row_ids * 1.7) + (col_ids * 2.9) + (frame_index * 0.85), 9.0) < 0.9
    ).astype(np_module.float32)
    sparkles *= 0.55

    grid_values = np_module.clip((trails * row_band * column_band) + sparkles, 0.0, 1.0)
    upsampled_grid = cv2_module.resize(
        grid_values.astype(np_module.float32),
        (width, height),
        interpolation=cv2_module.INTER_NEAREST,
    )

    y_coords = np_module.arange(height, dtype=np_module.float32).reshape(height, 1)
    x_coords = np_module.arange(width, dtype=np_module.float32).reshape(1, width)
    dy = np_module.mod(y_coords - (step_y / 2.0), step_y)
    dy = np_module.minimum(dy, step_y - dy)
    dx = np_module.mod(x_coords - (step_x / 2.0), step_x)
    dx = np_module.minimum(dx, step_x - dx)

    dot_kernel = np_module.exp(-((dx ** 2) + (dy ** 2)) / (2.0 * (1.18 ** 2)))
    row_line = np_module.exp(-(dy ** 2) / (2.0 * (0.42 ** 2)))
    column_line = np_module.exp(-(dx ** 2) / (2.0 * (0.55 ** 2)))

    dot_field = upsampled_grid * ((dot_kernel * 1.85) + (row_line * 0.22) + (column_line * 0.06))
    dot_glow = cv2_module.GaussianBlur(dot_field, (0, 0), 1.3)
    dot_core = cv2_module.GaussianBlur(dot_field, (0, 0), 0.35)

    return np_module.clip((dot_core * 1.65) + (dot_glow * 0.55), 0.0, 1.0)


def _apply_hologram_effect(
    cv2_module: object,
    np_module: object,
    frame: object,
    mask: object,
    *,
    frame_index: int,
    glow_strength: float,
    edge_strength: float,
    ghost_strength: float,
    scanline_alpha: float,
    scanline_spacing: int,
    subject_opacity: float,
    rng: object,
) -> object:
    height, width = frame.shape[:2]
    frame_float = frame.astype(np_module.float32) / 255.0
    luminance = cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2GRAY).astype(np_module.float32)
    luminance /= 255.0

    contrast_luma = cv2_module.equalizeHist((luminance * 255).astype(np_module.uint8))
    contrast_luma = contrast_luma.astype(np_module.float32) / 255.0
    contrast_luma = cv2_module.GaussianBlur(contrast_luma, (0, 0), 0.8)

    tint = np_module.array([1.00, 0.95, 0.56], dtype=np_module.float32)
    accent_tint = np_module.array([0.65, 0.95, 1.00], dtype=np_module.float32)
    cool_source = frame_float * np_module.array([1.10, 0.95, 0.42], dtype=np_module.float32)
    colored_luma = contrast_luma[..., None] * tint
    subject = np_module.clip((0.14 * cool_source) + (0.76 * colored_luma), 0.0, 1.0)
    subject *= mask[..., None]
    subject_surface = subject * (0.40 + (0.35 * contrast_luma[..., None]))

    glow = cv2_module.GaussianBlur(subject_surface, (0, 0), 6.0)
    aura_mask = cv2_module.GaussianBlur(mask, (0, 0), 16.0)
    aura_mask = np_module.clip(aura_mask - (mask * 0.16), 0.0, 1.0)

    edge_map = cv2_module.Canny((contrast_luma * 255).astype(np_module.uint8), 40, 110)
    edge_map = cv2_module.GaussianBlur(
        edge_map.astype(np_module.float32) / 255.0,
        (0, 0),
        1.5,
    )
    edge_map *= mask
    edge_color = edge_map[..., None] * np_module.array([1.0, 0.98, 0.86], dtype=np_module.float32)

    digital_rain = _build_digital_rain(
        cv2_module,
        np_module,
        height=height,
        width=width,
        frame_index=frame_index,
        mask=mask,
    )
    point_cloud = _build_point_cloud(
        cv2_module,
        np_module,
        contrast_luma,
        mask,
        frame_index=frame_index,
    )

    scanlines = _build_scanlines(
        np_module,
        height=height,
        width=width,
        frame_index=frame_index,
        spacing=scanline_spacing,
        alpha=scanline_alpha * 0.35,
    )

    ghost_x = int(round(2.0 * np_module.sin(frame_index * 0.27)))
    ghost_y = int(round(1.0 * np_module.cos(frame_index * 0.19)))
    ghost = np_module.roll(subject, shift=(ghost_y, ghost_x), axis=(0, 1))
    ghost = cv2_module.GaussianBlur(ghost, (0, 0), 1.2)
    glitch_overlay = _build_glitch_overlay(
        cv2_module,
        np_module,
        subject,
        mask,
        frame_index=frame_index,
    )

    row_noise = rng.normal(0.0, 0.012, size=(height, 1, 1)).astype(np_module.float32)
    row_noise = row_noise * np_module.array([0.08, 0.10, 0.12], dtype=np_module.float32)

    rows = np_module.arange(height, dtype=np_module.float32).reshape(height, 1)
    cols = np_module.arange(width, dtype=np_module.float32).reshape(1, width)
    background_dot_matrix = _build_dot_matrix_background(
        cv2_module,
        np_module,
        height=height,
        width=width,
        frame_index=frame_index,
    )
    ambient_rain = cv2_module.GaussianBlur(background_dot_matrix, (0, 0), 6.0)
    background_base = np_module.array(DEFAULT_BACKGROUND_BGR, dtype=np_module.float32) / 255.0
    ambient_background = ambient_rain[..., None] * np_module.array(
        [0.06, 0.08, 0.12],
        dtype=np_module.float32,
    )

    center_x = (cols - (width / 2.0)) / max(width, 1)
    center_y = (rows - (height * 0.55)) / max(height, 1)
    background_haze = np_module.exp(-((center_x ** 2) * 7.5) - ((center_y ** 2) * 10.0))
    background = background_base.reshape(1, 1, 3).copy()
    background = background + ambient_background + (
        background_haze[..., None] * np_module.array([0.05, 0.07, 0.12], dtype=np_module.float32)
    )
    background += background_dot_matrix[..., None] * np_module.array(
        [0.18, 0.30, 0.12],
        dtype=np_module.float32,
    )

    subject_field = np_module.clip(
        mask + (aura_mask * 0.65) + (digital_rain * 0.18),
        0.0,
        1.0,
    )
    background = background * (0.96 + (0.04 * scanlines[..., None]))
    background += row_noise * 0.45

    subject_layers = subject_surface * (subject_opacity * 0.46)
    subject_layers *= scanlines[..., None]
    subject_layers += glow * (glow_strength * 0.42)
    subject_layers += aura_mask[..., None] * tint * (glow_strength * 0.28)
    subject_layers += edge_color * (edge_strength * 1.55)
    subject_layers += digital_rain[..., None] * tint * 0.98
    subject_layers += point_cloud[..., None] * accent_tint * 1.28
    subject_layers += glitch_overlay
    subject_layers += ghost * (ghost_strength * 0.55)
    subject_layers += row_noise

    subject_layers = cv2_module.addWeighted(
        subject_layers,
        1.18,
        cv2_module.GaussianBlur(subject_layers, (0, 0), 0.9),
        -0.18,
        0.0,
    )

    hologram = background + (subject_layers * subject_field[..., None])

    return np_module.clip(hologram * 255.0, 0.0, 255.0).astype(np_module.uint8)


def generate_hologram_video(
    source_media_path: Union[Path, str],
    output_path: Union[Path, str],
    *,
    mask_threshold: int = DEFAULT_MASK_THRESHOLD,
    glow_strength: float = DEFAULT_GLOW_STRENGTH,
    edge_strength: float = DEFAULT_EDGE_STRENGTH,
    ghost_strength: float = DEFAULT_GHOST_STRENGTH,
    scanline_alpha: float = DEFAULT_SCANLINE_ALPHA,
    scanline_spacing: int = DEFAULT_SCANLINE_SPACING,
    subject_opacity: float = DEFAULT_SUBJECT_OPACITY,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Path:
    source_video = Path(source_media_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()

    _validate_source_video(source_video)
    if source_video == output_file:
        raise ValueError("The hologram stage requires a different output path than the input.")
    _validate_runtime_dependencies()

    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video for styling: {source_video}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Could not determine video size for {source_video}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    workspace_dir = Path(
        tempfile.mkdtemp(prefix=f"{output_file.stem}_style_", dir=output_file.parent)
    )
    staged_video_path = workspace_dir / output_file.name
    writer = cv2.VideoWriter(
        str(staged_video_path),
        _resolve_writer_fourcc(cv2, output_file),
        _resolve_frame_rate(capture, cv2),
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video writer for {output_file}")

    rng = np.random.default_rng(random_seed)
    previous_mask = None
    frame_index = 0

    try:
        while True:
            has_frame, frame = capture.read()
            if not has_frame:
                break

            mask = _extract_subject_mask(
                cv2,
                np,
                frame,
                previous_mask,
                mask_threshold=mask_threshold,
            )
            styled_frame = _apply_hologram_effect(
                cv2,
                np,
                frame,
                mask,
                frame_index=frame_index,
                glow_strength=glow_strength,
                edge_strength=edge_strength,
                ghost_strength=ghost_strength,
                scanline_alpha=scanline_alpha,
                scanline_spacing=scanline_spacing,
                subject_opacity=subject_opacity,
                rng=rng,
            )
            writer.write(styled_frame)
            previous_mask = mask
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    if frame_index == 0:
        staged_video_path.unlink(missing_ok=True)
        raise RuntimeError(f"No frames were decoded from {source_video}")

    try:
        muxed_output_path = workspace_dir / f"{output_file.stem}_muxed{output_file.suffix}"
        _run_audio_remux_subprocess(
            styled_video_path=staged_video_path,
            source_video_path=source_video,
            output_path=muxed_output_path,
        )
        shutil.move(str(muxed_output_path), str(output_file))
        return output_file
    finally:
        shutil.rmtree(workspace_dir, ignore_errors=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a blue hologram effect to a preprocessed face video. "
            "The input is expected to already have its background removed or "
            "rendered against near-black."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the already-matted face video.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path where the hologram-styled MP4 should be written.",
    )
    parser.add_argument(
        "--mask-threshold",
        default=DEFAULT_MASK_THRESHOLD,
        type=int,
        help="Brightness threshold used to isolate the face from the black background.",
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
        help="Darkening amount for the animated scanline overlay.",
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


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "__remux_audio__":
        raise SystemExit(_run_internal_remux_command(sys.argv[2:]))

    args = _parse_args()

    output_path = Path(args.output).expanduser().resolve()
    print(f"Generating hologram-styled MP4 at {output_path}...")
    generate_hologram_video(
        source_media_path=args.input,
        output_path=output_path,
        mask_threshold=args.mask_threshold,
        glow_strength=args.glow_strength,
        edge_strength=args.edge_strength,
        ghost_strength=args.ghost_strength,
        scanline_alpha=args.scanline_alpha,
        scanline_spacing=args.scanline_spacing,
        subject_opacity=args.subject_opacity,
    )
    print(f"Hologram MP4 output: {output_path}")


if __name__ == "__main__":
    main()
