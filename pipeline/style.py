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
DEFAULT_EFFECT_REFERENCE_EXTENT: Final = 320.0
DEFAULT_EFFECT_CANVAS_FILL_RATIO: Final = 0.72
DEFAULT_EFFECT_MASK_BOUNDS_THRESHOLD: Final = 0.12
DEFAULT_MAX_EFFECT_SCALE: Final = 4.5
DEFAULT_TINT_BGR: Final = (255, 230, 140)
DEFAULT_BACKGROUND_BGR: Final = (110, 42, 16)
DEFAULT_ALPHA_MASK_THRESHOLD: Final = 0.08
DEFAULT_ALPHA_SOFT_FLOOR: Final = 0.05
IMAGE_SOURCE_EXTENSIONS: Final = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_SOURCE_EXTENSIONS: Final = {".avi", ".m4v", ".mkv", ".mov", ".mp4"}
COMMON_REQUIRED_MODULE_PACKAGES: Final = {
    "cv2": "opencv-python-headless",
    "numpy": "numpy",
}
VIDEO_REQUIRED_MODULE_PACKAGES: Final = {
    "av": "av",
}


def _detect_media_type(media_path: Path, *, context: str) -> str:
    suffix = media_path.suffix.lower()
    if suffix in IMAGE_SOURCE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_SOURCE_EXTENSIONS:
        return "video"

    supported_extensions = ", ".join(
        sorted(IMAGE_SOURCE_EXTENSIONS | VIDEO_SOURCE_EXTENSIONS)
    )
    raise ValueError(
        f"Unsupported {context} format for {media_path}. Use one of: {supported_extensions}"
    )


def _validate_source_media(source_media: Path) -> str:
    if not source_media.is_file():
        raise FileNotFoundError(f"Style-stage source media was not found: {source_media}")
    return _detect_media_type(source_media, context="style-stage source media")


def _validate_output_media(output_path: Path, *, source_media_type: str) -> None:
    output_media_type = _detect_media_type(output_path, context="style-stage output media")
    if output_media_type != source_media_type:
        raise ValueError(
            "The hologram style stage requires matching input and output media types. "
            f"Received a {source_media_type} input but the output path resolves to a "
            f"{output_media_type} format: {output_path}"
        )


def _validate_runtime_dependencies(required_module_packages: dict[str, str]) -> None:
    missing_modules = [
        module_name
        for module_name in required_module_packages
        if importlib.util.find_spec(module_name) is None
    ]
    if missing_modules:
        rendered_packages = ", ".join(
            required_module_packages[module_name] for module_name in missing_modules
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


def _mask_bounds(np_module: object, mask: object) -> Optional[tuple[int, int, int, int]]:
    mask_points = np_module.column_stack(np_module.where(mask > 0))
    if mask_points.size == 0:
        return None

    y_coordinates = mask_points[:, 0]
    x_coordinates = mask_points[:, 1]
    return (
        int(x_coordinates.min()),
        int(y_coordinates.min()),
        int(x_coordinates.max()) + 1,
        int(y_coordinates.max()) + 1,
    )


def _constrain_alpha_portrait_mask(
    cv2_module: object,
    np_module: object,
    binary_mask: object,
) -> object:
    bounds = _mask_bounds(np_module, binary_mask)
    if bounds is None:
        return binary_mask

    mask_x1, mask_y1, mask_x2, mask_y2 = bounds
    mask_width = max(1, mask_x2 - mask_x1)
    mask_height = max(1, mask_y2 - mask_y1)
    center_x = mask_x1 + (mask_width / 2.0)
    refined_mask = binary_mask.copy()

    for row in range(mask_y1, mask_y2):
        vertical_position = (row - mask_y1) / max(mask_height, 1)
        if vertical_position < 0.10:
            half_width = mask_width * 0.54
        elif vertical_position < 0.46:
            progress = (vertical_position - 0.10) / 0.36
            half_width = mask_width * (0.54 - (0.06 * progress))
        elif vertical_position < 0.80:
            progress = (vertical_position - 0.46) / 0.34
            half_width = mask_width * (0.48 - (0.08 * progress))
        else:
            progress = min(1.0, max(0.0, (vertical_position - 0.80) / 0.20))
            half_width = mask_width * (0.40 - (0.14 * progress))

        left = max(0, min(binary_mask.shape[1], int(round(center_x - half_width))))
        right = max(left, min(binary_mask.shape[1], int(round(center_x + half_width))))
        refined_mask[row, :left] = 0
        refined_mask[row, right:] = 0

    return _largest_connected_component(cv2_module, np_module, refined_mask)


def _extract_subject_mask(
    cv2_module: object,
    np_module: object,
    frame: object,
    previous_mask: Optional[object],
    *,
    mask_threshold: int,
    subject_alpha: Optional[object] = None,
) -> object:
    if subject_alpha is not None:
        alpha_mask = np_module.clip(subject_alpha.astype(np_module.float32), 0.0, 1.0)
        binary_mask = np_module.where(
            alpha_mask > DEFAULT_ALPHA_MASK_THRESHOLD,
            255,
            0,
        ).astype(np_module.uint8)
    else:
        max_channel = frame.max(axis=2)
        binary_mask = np_module.where(max_channel > mask_threshold, 255, 0).astype(
            np_module.uint8
        )

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
    if subject_alpha is not None:
        alpha_energy = np_module.clip(
            (alpha_mask - DEFAULT_ALPHA_SOFT_FLOOR)
            / max(1.0 - DEFAULT_ALPHA_SOFT_FLOOR, 1e-6),
            0.0,
            1.0,
        )
        soft_mask = np_module.maximum(soft_mask, alpha_energy)
    else:
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


def _load_source_image(
    cv2_module: object,
    np_module: object,
    source_image: Path,
) -> tuple[object, Optional[object]]:
    image = cv2_module.imread(str(source_image), cv2_module.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read source image for styling: {source_image}")

    if len(image.shape) == 2:
        return cv2_module.cvtColor(image, cv2_module.COLOR_GRAY2BGR), None

    if len(image.shape) != 3:
        raise RuntimeError(f"Unsupported source image shape for styling: {source_image}")

    channel_count = image.shape[2]
    if channel_count == 3:
        return image, None

    if channel_count == 4:
        alpha_mask = image[:, :, 3].astype(np_module.float32) / 255.0
        color_channels = image[:, :, :3].astype(np_module.float32)
        premultiplied_bgr = np_module.clip(
            color_channels * alpha_mask[..., None],
            0.0,
            255.0,
        ).astype(np_module.uint8)
        return premultiplied_bgr, alpha_mask

    raise RuntimeError(
        f"Unsupported source image channel count for styling: {channel_count} in {source_image}"
    )


def _style_frame(
    cv2_module: object,
    np_module: object,
    frame: object,
    previous_mask: Optional[object],
    *,
    frame_index: int,
    mask_threshold: int,
    glow_strength: float,
    edge_strength: float,
    ghost_strength: float,
    scanline_alpha: float,
    scanline_spacing: int,
    subject_opacity: float,
    subject_alpha: Optional[object],
    rng: object,
) -> tuple[object, object]:
    mask = _extract_subject_mask(
        cv2_module,
        np_module,
        frame,
        previous_mask,
        mask_threshold=mask_threshold,
        subject_alpha=subject_alpha,
    )
    styled_frame = _apply_hologram_effect(
        cv2_module,
        np_module,
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
    return styled_frame, mask


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


def _scale_effect_radius(base_radius: float, effect_scale: float, *, minimum: float = 0.01) -> float:
    return max(minimum, base_radius * effect_scale)


def _scale_effect_step(base_step: int, effect_scale: float) -> int:
    return max(1, int(round(base_step * effect_scale)))


def _resolve_effect_scale(np_module: object, mask: object) -> float:
    height, width = mask.shape
    effective_extent = float(min(height, width)) * DEFAULT_EFFECT_CANVAS_FILL_RATIO
    subject_bounds = _mask_bounds(
        np_module,
        np_module.where(mask > DEFAULT_EFFECT_MASK_BOUNDS_THRESHOLD, 255, 0).astype(
            np_module.uint8
        ),
    )
    if subject_bounds is not None:
        x1, y1, x2, y2 = subject_bounds
        effective_extent = max(
            effective_extent,
            float(max(x2 - x1, y2 - y1)),
        )

    return min(
        DEFAULT_MAX_EFFECT_SCALE,
        max(1.0, effective_extent / DEFAULT_EFFECT_REFERENCE_EXTENT),
    )


def _build_reference_dot_layer(
    cv2_module: object,
    np_module: object,
    contrast_luma: object,
    mask: object,
    edge_map: object,
    warm_map: object,
    *,
    effect_scale: float,
    frame_index: int,
) -> object:
    height, width = mask.shape
    step_x = _scale_effect_step(4, effect_scale)
    step_y = _scale_effect_step(4, effect_scale)
    dot_sigma = _scale_effect_radius(1.18, effect_scale)
    dot_blur_sigma = _scale_effect_radius(0.64, effect_scale)
    highlight_sigma = _scale_effect_radius(0.9, effect_scale)
    grid_width = max(1, (width + step_x - 1) // step_x)
    grid_height = max(1, (height + step_y - 1) // step_y)

    sampled_mask = cv2_module.resize(
        mask.astype(np_module.float32),
        (grid_width, grid_height),
        interpolation=cv2_module.INTER_AREA,
    )
    sampled_luma = cv2_module.resize(
        contrast_luma.astype(np_module.float32),
        (grid_width, grid_height),
        interpolation=cv2_module.INTER_AREA,
    )
    sampled_edges = cv2_module.resize(
        edge_map.astype(np_module.float32),
        (grid_width, grid_height),
        interpolation=cv2_module.INTER_AREA,
    )
    sampled_warm = cv2_module.resize(
        warm_map.astype(np_module.float32),
        (grid_width, grid_height),
        interpolation=cv2_module.INTER_AREA,
    )

    temporal_phase = frame_index * 0.028
    row_ids = np_module.arange(grid_height, dtype=np_module.float32).reshape(-1, 1)
    col_ids = np_module.arange(grid_width, dtype=np_module.float32).reshape(1, -1)
    noise_a = np_module.mod(
        np_module.sin((col_ids * 12.9898) + (row_ids * 78.233) + temporal_phase)
        * 43758.5453,
        1.0,
    ).astype(np_module.float32)
    noise_b = np_module.mod(
        np_module.sin((col_ids * 4.123) + (row_ids * 91.731) + (temporal_phase * 0.72))
        * 24634.6345,
        1.0,
    ).astype(np_module.float32)
    noise_c = np_module.mod(
        np_module.sin((col_ids * 31.221) + (row_ids * 17.173) + (temporal_phase * 0.44))
        * 17321.372,
        1.0,
    ).astype(np_module.float32)
    color_wave_a = 0.5 + 0.5 * np_module.sin(
        (col_ids * 0.31) + (row_ids * 0.17) + (temporal_phase * 0.20)
    )
    color_wave_b = 0.5 + 0.5 * np_module.sin(
        (col_ids * 0.19) - (row_ids * 0.23) + (temporal_phase * 0.14) + 1.3
    )

    dense_field = np_module.clip(
        ((sampled_luma ** 1.24) * 0.68)
        + (sampled_edges * 0.30)
        + (sampled_mask * 0.24),
        0.0,
        1.0,
    )
    dense_field *= 0.42 + (0.58 * sampled_mask)
    dense_field *= 0.88 + (0.22 * noise_a)
    dense_field = np_module.clip(dense_field, 0.0, 1.0)

    blue_weight_grid = np_module.clip(
        0.70
        + (0.26 * (sampled_luma ** 1.7))
        + (0.18 * color_wave_a)
        + (0.10 * noise_a)
        - (0.08 * sampled_warm),
        0.30,
        1.38,
    ).astype(np_module.float32)
    green_weight_grid = np_module.clip(
        0.54
        + (0.18 * color_wave_b)
        + (0.16 * noise_b)
        + (0.12 * (1.0 - sampled_luma))
        + (0.08 * sampled_edges),
        0.22,
        1.24,
    ).astype(np_module.float32)
    red_weight_grid = np_module.clip(
        0.10
        + (0.42 * sampled_warm)
        + (0.18 * (1.0 - color_wave_a))
        + (0.14 * noise_c)
        + (0.06 * sampled_edges),
        0.04,
        1.08,
    ).astype(np_module.float32)

    upsampled_field = cv2_module.resize(
        dense_field.astype(np_module.float32),
        (width, height),
        interpolation=cv2_module.INTER_NEAREST,
    )
    noise_a = cv2_module.resize(
        noise_a.astype(np_module.float32),
        (width, height),
        interpolation=cv2_module.INTER_NEAREST,
    )
    noise_b = cv2_module.resize(
        noise_b.astype(np_module.float32),
        (width, height),
        interpolation=cv2_module.INTER_NEAREST,
    )
    noise_c = cv2_module.resize(
        noise_c.astype(np_module.float32),
        (width, height),
        interpolation=cv2_module.INTER_NEAREST,
    )
    blue_weight = cv2_module.resize(
        blue_weight_grid,
        (width, height),
        interpolation=cv2_module.INTER_NEAREST,
    )
    green_weight = cv2_module.resize(
        green_weight_grid,
        (width, height),
        interpolation=cv2_module.INTER_NEAREST,
    )
    red_weight = cv2_module.resize(
        red_weight_grid,
        (width, height),
        interpolation=cv2_module.INTER_NEAREST,
    )

    y_coords = np_module.arange(height, dtype=np_module.float32).reshape(height, 1)
    x_coords = np_module.arange(width, dtype=np_module.float32).reshape(1, width)
    dy = np_module.mod(y_coords - (step_y / 2.0), step_y)
    dy = np_module.minimum(dy, step_y - dy)
    dx = np_module.mod(x_coords - (step_x / 2.0), step_x)
    dx = np_module.minimum(dx, step_x - dx)
    dot_kernel = np_module.exp(-((dx ** 2) + (dy ** 2)) / (2.0 * (dot_sigma ** 2)))

    dot_field = upsampled_field * dot_kernel
    dot_field = np_module.clip(
        (dot_field * 1.06)
        + (cv2_module.GaussianBlur(dot_field, (0, 0), dot_blur_sigma) * 0.22),
        0.0,
        1.0,
    )

    highlight_core = cv2_module.GaussianBlur(
        (contrast_luma ** 6.2) * mask,
        (0, 0),
        highlight_sigma,
    )
    highlight_core = np_module.clip(highlight_core, 0.0, 1.0)

    blue = np_module.clip(
        (dot_field * blue_weight) + (highlight_core * 0.16),
        0.0,
        1.0,
    )
    green = np_module.clip(
        (np_module.roll(dot_field, shift=1, axis=1) * green_weight)
        + (highlight_core * 0.14),
        0.0,
        1.0,
    )
    red = np_module.clip(
        (np_module.roll(dot_field, shift=-1, axis=1) * red_weight)
        + (np_module.roll(highlight_core, shift=1, axis=1) * 0.06)
        + (edge_map * 0.06),
        0.0,
        1.0,
    )
    return np_module.stack([blue, green, red], axis=2)


def _build_aura_particles(
    cv2_module: object,
    np_module: object,
    mask: object,
    edge_map: object,
    *,
    effect_scale: float,
    frame_index: int,
) -> object:
    height, width = mask.shape
    aura_shell = cv2_module.GaussianBlur(
        mask.astype(np_module.float32),
        (0, 0),
        _scale_effect_radius(14.0, effect_scale),
    )
    aura_shell = np_module.clip(aura_shell - (mask * 0.22), 0.0, 1.0)

    step = _scale_effect_step(5, effect_scale)
    particle_sigma = _scale_effect_radius(0.86, effect_scale)
    particle_blur_sigma = _scale_effect_radius(1.0, effect_scale)
    grid_width = max(1, (width + step - 1) // step)
    grid_height = max(1, (height + step - 1) // step)
    sampled_shell = cv2_module.resize(
        aura_shell.astype(np_module.float32),
        (grid_width, grid_height),
        interpolation=cv2_module.INTER_AREA,
    )
    sampled_edges = cv2_module.resize(
        edge_map.astype(np_module.float32),
        (grid_width, grid_height),
        interpolation=cv2_module.INTER_AREA,
    )

    row_ids = np_module.arange(grid_height, dtype=np_module.float32).reshape(-1, 1)
    col_ids = np_module.arange(grid_width, dtype=np_module.float32).reshape(1, -1)
    temporal_phase = frame_index * 0.020
    twinkle = np_module.mod(
        np_module.sin((col_ids * 15.134) + (row_ids * 61.79) + temporal_phase)
        * 42123.12,
        1.0,
    ).astype(np_module.float32)
    clusters = 0.5 + 0.5 * np_module.sin(
        (col_ids * 0.83) - (row_ids * 0.52) + (temporal_phase * 0.25)
    )
    particle_probability = np_module.clip(
        sampled_shell * (0.20 + (0.80 * clusters))
        + (sampled_edges * 0.08),
        0.0,
        1.0,
    )
    active_particles = (twinkle > (0.94 - (particle_probability * 0.18))).astype(
        np_module.float32
    )
    particle_values = active_particles * particle_probability * (0.30 + (0.70 * twinkle))

    upsampled_particles = cv2_module.resize(
        particle_values.astype(np_module.float32),
        (width, height),
        interpolation=cv2_module.INTER_NEAREST,
    )

    y_coords = np_module.arange(height, dtype=np_module.float32).reshape(height, 1)
    x_coords = np_module.arange(width, dtype=np_module.float32).reshape(1, width)
    dy = np_module.mod(y_coords - (step / 2.0), step)
    dy = np_module.minimum(dy, step - dy)
    dx = np_module.mod(x_coords - (step / 2.0), step)
    dx = np_module.minimum(dx, step - dx)
    particle_kernel = np_module.exp(
        -((dx ** 2) + (dy ** 2)) / (2.0 * (particle_sigma ** 2))
    )
    particle_field = upsampled_particles * particle_kernel
    particle_field = np_module.clip(
        (particle_field * 1.45)
        + (cv2_module.GaussianBlur(particle_field, (0, 0), particle_blur_sigma) * 0.45),
        0.0,
        1.0,
    )
    blue = particle_field * 0.95
    green = np_module.roll(particle_field, shift=1, axis=1) * 0.86
    red = np_module.roll(particle_field, shift=2, axis=1) * 0.58
    return np_module.stack([blue, green, red], axis=2)


def _build_edge_dispersion(
    cv2_module: object,
    np_module: object,
    mask: object,
    edge_map: object,
    warm_map: object,
    *,
    effect_scale: float,
    frame_index: int,
) -> object:
    height, width = mask.shape
    inner_blur = cv2_module.GaussianBlur(
        mask.astype(np_module.float32),
        (0, 0),
        _scale_effect_radius(4.8, effect_scale),
    )
    outer_blur = cv2_module.GaussianBlur(
        mask.astype(np_module.float32),
        (0, 0),
        _scale_effect_radius(18.0, effect_scale),
    )
    edge_shell = np_module.clip(outer_blur - (inner_blur * 0.72), 0.0, 1.0)
    perimeter_energy = np_module.clip((edge_map * 0.28) + (edge_shell * 1.12), 0.0, 1.0)

    step = _scale_effect_step(6, effect_scale)
    particle_sigma = _scale_effect_radius(1.40, effect_scale)
    edge_blur_sigma = _scale_effect_radius(2.2, effect_scale)
    grid_width = max(1, (width + step - 1) // step)
    grid_height = max(1, (height + step - 1) // step)
    sampled_energy = cv2_module.resize(
        perimeter_energy.astype(np_module.float32),
        (grid_width, grid_height),
        interpolation=cv2_module.INTER_AREA,
    )
    sampled_warm = cv2_module.resize(
        warm_map.astype(np_module.float32),
        (grid_width, grid_height),
        interpolation=cv2_module.INTER_AREA,
    )

    row_ids = np_module.arange(grid_height, dtype=np_module.float32).reshape(-1, 1)
    col_ids = np_module.arange(grid_width, dtype=np_module.float32).reshape(1, -1)
    temporal_phase = frame_index * 0.014
    noise_a = np_module.mod(
        np_module.sin((col_ids * 8.312) + (row_ids * 43.17) + temporal_phase) * 27813.71,
        1.0,
    ).astype(np_module.float32)
    noise_b = np_module.mod(
        np_module.sin((col_ids * 17.91) + (row_ids * 11.43) + (temporal_phase * 0.68)) * 19641.11,
        1.0,
    ).astype(np_module.float32)
    activation = (noise_a > (0.92 - (sampled_energy * 0.20))).astype(np_module.float32)
    edge_values = sampled_energy * (0.22 + (0.48 * noise_b)) * (0.35 + (0.65 * activation))

    upsampled_edge_values = cv2_module.resize(
        edge_values.astype(np_module.float32),
        (width, height),
        interpolation=cv2_module.INTER_NEAREST,
    )
    sampled_warm = cv2_module.resize(
        sampled_warm.astype(np_module.float32),
        (width, height),
        interpolation=cv2_module.INTER_NEAREST,
    )

    y_coords = np_module.arange(height, dtype=np_module.float32).reshape(height, 1)
    x_coords = np_module.arange(width, dtype=np_module.float32).reshape(1, width)
    dy = np_module.mod(y_coords - (step / 2.0), step)
    dy = np_module.minimum(dy, step - dy)
    dx = np_module.mod(x_coords - (step / 2.0), step)
    dx = np_module.minimum(dx, step - dx)
    particle_kernel = np_module.exp(
        -((dx ** 2) + (dy ** 2)) / (2.0 * (particle_sigma ** 2))
    )
    edge_particles = upsampled_edge_values * particle_kernel
    edge_particles = np_module.clip(
        (edge_particles * 1.58)
        + (cv2_module.GaussianBlur(edge_particles, (0, 0), edge_blur_sigma) * 0.78),
        0.0,
        1.0,
    )

    blue = np_module.clip(edge_particles * (0.92 - (0.16 * sampled_warm)), 0.0, 1.0)
    green = np_module.clip(
        np_module.roll(edge_particles, shift=1, axis=1) * (0.76 + (0.10 * (1.0 - sampled_warm))),
        0.0,
        1.0,
    )
    red = np_module.clip(
        np_module.roll(edge_particles, shift=-1, axis=1) * (0.18 + (0.44 * sampled_warm)),
        0.0,
        1.0,
    )
    return np_module.stack([blue, green, red], axis=2)


def _build_reference_background(
    np_module: object,
    *,
    height: int,
    width: int,
) -> object:
    y_coords = np_module.arange(height, dtype=np_module.float32).reshape(height, 1)
    x_coords = np_module.arange(width, dtype=np_module.float32).reshape(1, width)
    center_x = (x_coords - (width / 2.0)) / max(width, 1)
    center_y = (y_coords - (height * 0.48)) / max(height, 1)
    radial_haze = np_module.exp(-((center_x ** 2) * 5.0) - ((center_y ** 2) * 7.0))
    vertical_falloff = np_module.clip(1.15 - (y_coords / max(height, 1)), 0.20, 1.15)
    vignette = np_module.clip(
        1.0 - (((center_x ** 2) * 0.85) + ((center_y ** 2) * 1.25)),
        0.0,
        1.0,
    )

    background = np_module.zeros((height, width, 3), dtype=np_module.float32)
    background += radial_haze[..., None] * np_module.array([0.020, 0.025, 0.032], dtype=np_module.float32)
    background += (radial_haze * vertical_falloff)[..., None] * np_module.array(
        [0.008, 0.011, 0.014],
        dtype=np_module.float32,
    )
    return np_module.clip(background * vignette[..., None], 0.0, 1.0)


def _build_portrait_focus(np_module: object, mask: object) -> object:
    height, width = mask.shape
    coords = np_module.argwhere(mask > 0.12)
    if coords.size == 0:
        return np_module.ones((height, width), dtype=np_module.float32)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    center_x = float(x0 + ((x1 - x0) * 0.50))
    center_y = float(y0 + ((y1 - y0) * 0.46))
    radius_x = max(float((x1 - x0) * 0.46), width * 0.24)
    radius_y = max(float((y1 - y0) * 0.54), height * 0.28)

    y_coords = np_module.arange(height, dtype=np_module.float32).reshape(height, 1)
    x_coords = np_module.arange(width, dtype=np_module.float32).reshape(1, width)
    focus = np_module.exp(
        -(
            (((x_coords - center_x) ** 2) / (2.0 * (radius_x ** 2)))
            + (((y_coords - center_y) ** 2) / (2.0 * (radius_y ** 2)))
        )
    )
    focus = np_module.clip(focus * 1.08, 0.0, 1.0)
    return np_module.maximum(mask * 0.55, focus.astype(np_module.float32))


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
    effect_scale = _resolve_effect_scale(np_module, mask)
    frame_float = frame.astype(np_module.float32) / 255.0
    luminance = cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2GRAY).astype(np_module.float32)
    luminance /= 255.0

    contrast_luma = cv2_module.equalizeHist((luminance * 255).astype(np_module.uint8))
    contrast_luma = contrast_luma.astype(np_module.float32) / 255.0
    contrast_luma = (luminance * 0.58) + (contrast_luma * 0.42)
    contrast_luma = cv2_module.GaussianBlur(
        contrast_luma,
        (0, 0),
        _scale_effect_radius(0.8, effect_scale),
    )
    contrast_luma = np_module.clip(contrast_luma, 0.0, 1.0) ** 1.12
    soft_subject = cv2_module.GaussianBlur(
        mask.astype(np_module.float32),
        (0, 0),
        _scale_effect_radius(4.8, effect_scale),
    )
    feather_subject = cv2_module.GaussianBlur(
        mask.astype(np_module.float32),
        (0, 0),
        _scale_effect_radius(9.0, effect_scale),
    )
    outer_subject = cv2_module.GaussianBlur(
        mask.astype(np_module.float32),
        (0, 0),
        _scale_effect_radius(18.0, effect_scale),
    )

    aura_mask = cv2_module.GaussianBlur(
        mask,
        (0, 0),
        _scale_effect_radius(16.0, effect_scale),
    )
    aura_mask = np_module.clip(aura_mask - (mask * 0.08), 0.0, 1.0)

    edge_map = cv2_module.Canny((contrast_luma * 255).astype(np_module.uint8), 34, 92)
    edge_map = cv2_module.GaussianBlur(
        edge_map.astype(np_module.float32) / 255.0,
        (0, 0),
        _scale_effect_radius(1.0, effect_scale),
    )
    edge_map *= mask
    edge_color = edge_map[..., None] * np_module.array([1.0, 0.96, 0.72], dtype=np_module.float32)
    portrait_focus = _build_portrait_focus(np_module, mask)
    subject_presence = np_module.clip(
        (soft_subject * 0.50) + (feather_subject * 0.26) + (portrait_focus * 0.24),
        0.0,
        1.0,
    )
    subject_blend = np_module.clip((soft_subject * 0.56) + (feather_subject * 0.44), 0.0, 1.0)
    edge_falloff = np_module.clip(outer_subject - (soft_subject * 0.62), 0.0, 1.0)
    warm_accent = np_module.clip(
        frame_float[..., 2] - (frame_float[..., 0] * 0.55),
        0.0,
        1.0,
    )
    warm_accent *= mask
    warm_accent = cv2_module.GaussianBlur(
        warm_accent,
        (0, 0),
        _scale_effect_radius(1.1, effect_scale),
    )

    dot_layer = _build_reference_dot_layer(
        cv2_module,
        np_module,
        contrast_luma,
        feather_subject,
        edge_map,
        warm_accent,
        effect_scale=effect_scale,
        frame_index=frame_index,
    )
    aura_particles = _build_aura_particles(
        cv2_module,
        np_module,
        mask,
        edge_map,
        effect_scale=effect_scale,
        frame_index=frame_index,
    )
    edge_dispersion = _build_edge_dispersion(
        cv2_module,
        np_module,
        mask,
        edge_map,
        warm_accent,
        effect_scale=effect_scale,
        frame_index=frame_index,
    )
    display_texture = 1.0

    source_reflection = frame_float * feather_subject[..., None]
    source_reflection *= np_module.array([0.06, 0.04, 0.04], dtype=np_module.float32)
    source_reflection = cv2_module.GaussianBlur(
        source_reflection,
        (0, 0),
        _scale_effect_radius(0.8, effect_scale),
    )

    core_glow = cv2_module.GaussianBlur(
        dot_layer,
        (0, 0),
        _scale_effect_radius(1.2, effect_scale),
    )
    halo_glow = cv2_module.GaussianBlur(
        dot_layer,
        (0, 0),
        _scale_effect_radius(3.8, effect_scale),
    )
    edge_haze = cv2_module.GaussianBlur(
        edge_dispersion,
        (0, 0),
        _scale_effect_radius(4.0, effect_scale),
    )
    highlight_glow = cv2_module.GaussianBlur(
        (contrast_luma ** 6.0) * mask * portrait_focus,
        (0, 0),
        _scale_effect_radius(1.9, effect_scale),
    )
    highlight_glow = highlight_glow[..., None] * np_module.array(
        [1.0, 1.0, 1.0],
        dtype=np_module.float32,
    )
    white_core = cv2_module.GaussianBlur(
        (contrast_luma ** 8.0) * mask * portrait_focus,
        (0, 0),
        _scale_effect_radius(0.7, effect_scale),
    )[..., None]

    ghost_x = int(
        round(
            _scale_effect_radius(1.0, effect_scale)
            + (_scale_effect_radius(1.5, effect_scale) * np_module.sin(frame_index * 0.08))
        )
    )
    ghost_y = int(
        round(_scale_effect_radius(0.6, effect_scale) * np_module.cos(frame_index * 0.06))
    )
    chromatic_ghost = np_module.roll(dot_layer, shift=(ghost_y, ghost_x), axis=(0, 1))
    chromatic_ghost = cv2_module.GaussianBlur(
        chromatic_ghost,
        (0, 0),
        _scale_effect_radius(0.9, effect_scale),
    )

    background = _build_reference_background(
        np_module,
        height=height,
        width=width,
    )
    background += halo_glow * np_module.array([0.008, 0.010, 0.014], dtype=np_module.float32)
    background += edge_haze * 0.30
    background += edge_dispersion * 0.06
    background += edge_falloff[..., None] * np_module.array([0.010, 0.014, 0.020], dtype=np_module.float32)
    background += aura_particles * 0.10

    film_grain = rng.normal(0.0, 0.006, size=(height, width, 1)).astype(np_module.float32)
    background += film_grain * np_module.array([0.006, 0.008, 0.010], dtype=np_module.float32)

    shading = (0.24 + (0.76 * (contrast_luma ** 0.90)))[..., None]
    subject_layers = dot_layer * (subject_opacity * 0.92) * shading
    subject_layers *= (0.55 + (0.45 * subject_presence))[..., None]
    subject_layers *= subject_blend[..., None]
    subject_layers *= display_texture
    subject_layers += core_glow * (0.10 + (glow_strength * 0.36)) * subject_blend[..., None]
    subject_layers += halo_glow * (0.02 + (glow_strength * 0.12)) * feather_subject[..., None]
    subject_layers += aura_mask[..., None] * np_module.array([0.025, 0.035, 0.050], dtype=np_module.float32)
    subject_layers += edge_color * (edge_strength * 0.08)
    subject_layers += edge_dispersion * (0.26 + (edge_strength * 0.48))
    subject_layers += edge_haze * (0.06 + (edge_strength * 0.16))
    subject_layers += highlight_glow * (0.10 + (glow_strength * 0.28))
    subject_layers += white_core * (0.10 + (glow_strength * 0.26))
    subject_layers += chromatic_ghost * (ghost_strength * 0.20)
    subject_layers += warm_accent[..., None] * np_module.array([0.01, 0.04, 0.14], dtype=np_module.float32)
    subject_layers += aura_particles * 0.22
    subject_layers += source_reflection * 0.05

    subject_layers = cv2_module.addWeighted(
        subject_layers,
        1.01,
        cv2_module.GaussianBlur(
            subject_layers,
            (0, 0),
            _scale_effect_radius(0.65, effect_scale),
        ),
        -0.01,
        0.0,
    )

    hologram = background + subject_layers

    return np_module.clip(hologram * 255.0, 0.0, 255.0).astype(np_module.uint8)


def _generate_hologram_image(
    source_image: Path,
    output_file: Path,
    *,
    mask_threshold: int,
    glow_strength: float,
    edge_strength: float,
    ghost_strength: float,
    scanline_alpha: float,
    scanline_spacing: int,
    subject_opacity: float,
    random_seed: int,
) -> Path:
    import cv2
    import numpy as np

    frame, subject_alpha = _load_source_image(cv2, np, source_image)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(random_seed)
    styled_frame, _ = _style_frame(
        cv2,
        np,
        frame,
        None,
        frame_index=0,
        mask_threshold=mask_threshold,
        glow_strength=glow_strength,
        edge_strength=edge_strength,
        ghost_strength=ghost_strength,
        scanline_alpha=scanline_alpha,
        scanline_spacing=scanline_spacing,
        subject_opacity=subject_opacity,
        subject_alpha=subject_alpha,
        rng=rng,
    )

    if not cv2.imwrite(str(output_file), styled_frame):
        raise RuntimeError(f"Could not write styled image to {output_file}")
    return output_file


def _generate_hologram_video(
    source_video: Path,
    output_file: Path,
    *,
    mask_threshold: int,
    glow_strength: float,
    edge_strength: float,
    ghost_strength: float,
    scanline_alpha: float,
    scanline_spacing: int,
    subject_opacity: float,
    random_seed: int,
) -> Path:
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

            styled_frame, previous_mask = _style_frame(
                cv2,
                np,
                frame,
                previous_mask,
                frame_index=frame_index,
                mask_threshold=mask_threshold,
                glow_strength=glow_strength,
                edge_strength=edge_strength,
                ghost_strength=ghost_strength,
                scanline_alpha=scanline_alpha,
                scanline_spacing=scanline_spacing,
                subject_opacity=subject_opacity,
                subject_alpha=None,
                rng=rng,
            )
            writer.write(styled_frame)
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
    source_media = Path(source_media_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()

    source_media_type = _validate_source_media(source_media)
    _validate_output_media(output_file, source_media_type=source_media_type)
    if source_media == output_file:
        raise ValueError("The hologram stage requires a different output path than the input.")

    required_module_packages = dict(COMMON_REQUIRED_MODULE_PACKAGES)
    if source_media_type == "video":
        required_module_packages.update(VIDEO_REQUIRED_MODULE_PACKAGES)
    _validate_runtime_dependencies(required_module_packages)

    if source_media_type == "image":
        return _generate_hologram_image(
            source_media,
            output_file,
            mask_threshold=mask_threshold,
            glow_strength=glow_strength,
            edge_strength=edge_strength,
            ghost_strength=ghost_strength,
            scanline_alpha=scanline_alpha,
            scanline_spacing=scanline_spacing,
            subject_opacity=subject_opacity,
            random_seed=random_seed,
        )

    return _generate_hologram_video(
        source_media,
        output_file,
        mask_threshold=mask_threshold,
        glow_strength=glow_strength,
        edge_strength=edge_strength,
        ghost_strength=ghost_strength,
        scanline_alpha=scanline_alpha,
        scanline_spacing=scanline_spacing,
        subject_opacity=subject_opacity,
        random_seed=random_seed,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a luminous dot-matrix portrait effect to a preprocessed face "
            "image or video. The input is expected to already have its background "
            "removed, rendered against near-black, or stored with transparency."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the already-isolated face image or video.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Path where the hologram-styled image or video should be written. "
            "Use an image extension for image input or a video extension for video input."
        ),
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
    print(f"Generating hologram-styled output at {output_path}...")
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
    print(f"Hologram output: {output_path}")


if __name__ == "__main__":
    main()
