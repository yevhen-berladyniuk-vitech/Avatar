from __future__ import annotations

import argparse
import importlib.util
import math
import os
from pathlib import Path
from typing import Final, Union

SUPPORTED_IMAGE_EXTENSIONS: Final = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".webp",
}
REQUIRED_MODULE_PACKAGES: Final = {
    "cv2": "opencv-python-headless",
    "numpy": "numpy",
    "onnxruntime": "onnxruntime",
    "PIL": "pillow",
    "rembg": "rembg",
}
CASCADE_FILENAMES: Final = (
    "haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_alt.xml",
    "haarcascade_frontalface_alt2.xml",
    "haarcascade_profileface.xml",
)
DEFAULT_PADDING_RATIO: Final = 0.08
DEFAULT_MAX_DETECTION_EDGE: Final = 1280
DEFAULT_MIN_FACE_SIZE: Final = 24
DEFAULT_SEGMENTATION_MAX_EDGE: Final = 1024
DEFAULT_MASK_BOUNDS_THRESHOLD: Final = 16
DEFAULT_ALPHA_TRIM: Final = 24
DEFAULT_SUBJECT_MASK_THRESHOLD: Final = 96
DEFAULT_HEAD_LIMIT_THRESHOLD: Final = 24
REMBG_MODEL_NAME: Final = "u2net_human_seg"

_REMBG_SESSION: object | None = None


def _validate_source_image(source_image: Path) -> None:
    if not source_image.is_file():
        raise FileNotFoundError(f"Source image was not found: {source_image}")

    if source_image.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        supported_extensions = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise ValueError(
            f"Unsupported image format for {source_image}. Use one of: {supported_extensions}"
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
            "The face crop utility requires additional Python packages in the current "
            f"environment. Install: {rendered_packages}"
        )


def _rembg_available() -> bool:
    return importlib.util.find_spec("rembg") is not None


def _load_image(cv2_module: object, source_image: Path) -> object:
    image = cv2_module.imread(str(source_image), cv2_module.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read source image: {source_image}")
    return image


def _candidate_cascade_paths(cv2_module: object) -> list[Path]:
    cascade_root = Path(cv2_module.data.haarcascades)
    return [
        cascade_root / cascade_name
        for cascade_name in CASCADE_FILENAMES
        if (cascade_root / cascade_name).is_file()
    ]


def _resize_for_detection(cv2_module: object, image: object) -> tuple[object, float]:
    height, width = image.shape[:2]
    longest_edge = max(height, width)
    if longest_edge <= DEFAULT_MAX_DETECTION_EDGE:
        return image, 1.0

    scale = DEFAULT_MAX_DETECTION_EDGE / float(longest_edge)
    resized_image = cv2_module.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2_module.INTER_AREA,
    )
    return resized_image, scale


def _prepare_detection_variants(cv2_module: object, gray_image: object) -> list[tuple[object, bool]]:
    equalized_image = cv2_module.equalizeHist(gray_image)
    flipped_gray = cv2_module.flip(gray_image, 1)
    flipped_equalized = cv2_module.equalizeHist(flipped_gray)
    return [
        (gray_image, False),
        (equalized_image, False),
        (flipped_gray, True),
        (flipped_equalized, True),
    ]


def _detection_score(image_shape: tuple[int, int], face_box: tuple[int, int, int, int]) -> float:
    image_height, image_width = image_shape
    x, y, width, height = face_box
    area = float(width * height)

    face_center_x = x + (width / 2.0)
    face_center_y = y + (height / 2.0)
    image_center_x = image_width / 2.0
    image_center_y = image_height / 2.0

    normalized_offset = math.hypot(
        (face_center_x - image_center_x) / max(image_width, 1),
        (face_center_y - image_center_y) / max(image_height, 1),
    )
    return area / (1.0 + normalized_offset)


def _detect_primary_face(cv2_module: object, image: object) -> tuple[int, int, int, int]:
    gray_image = cv2_module.cvtColor(image, cv2_module.COLOR_BGR2GRAY)
    detection_image, detection_scale = _resize_for_detection(cv2_module, gray_image)

    original_height, original_width = gray_image.shape[:2]
    resized_height, resized_width = detection_image.shape[:2]
    min_face_size = max(
        DEFAULT_MIN_FACE_SIZE,
        int(round(min(resized_height, resized_width) * 0.08)),
    )

    best_face_box: tuple[int, int, int, int] | None = None
    best_score = -1.0

    for cascade_path in _candidate_cascade_paths(cv2_module):
        classifier = cv2_module.CascadeClassifier(str(cascade_path))
        if classifier.empty():
            continue

        for candidate_image, mirrored in _prepare_detection_variants(cv2_module, detection_image):
            detections = classifier.detectMultiScale(
                candidate_image,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(min_face_size, min_face_size),
            )
            for detection in detections:
                x, y, width, height = map(int, detection)
                if mirrored:
                    x = resized_width - x - width

                if detection_scale != 1.0:
                    x = int(round(x / detection_scale))
                    y = int(round(y / detection_scale))
                    width = int(round(width / detection_scale))
                    height = int(round(height / detection_scale))

                x = max(0, min(x, original_width - 1))
                y = max(0, min(y, original_height - 1))
                width = max(1, min(width, original_width - x))
                height = max(1, min(height, original_height - y))
                face_box = (x, y, width, height)
                face_score = _detection_score((original_height, original_width), face_box)
                if face_score > best_score:
                    best_face_box = face_box
                    best_score = face_score

    if best_face_box is None:
        raise RuntimeError(
            "No face was detected in the input image. Try a clearer front-facing photo "
            "or reduce occlusion around the face."
        )

    return best_face_box


def _expand_bounds(
    image_shape: tuple[int, int],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    pad_x: int = 0,
    pad_y: int = 0,
) -> tuple[int, int, int, int]:
    image_height, image_width = image_shape
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(image_width, x2 + pad_x),
        min(image_height, y2 + pad_y),
    )


def _clamp_square_crop(
    image_width: int,
    image_height: int,
    center_x: float,
    center_y: float,
    side_length: int,
) -> tuple[int, int, int, int]:
    side_length = max(1, side_length)
    x1 = int(round(center_x - (side_length / 2.0)))
    y1 = int(round(center_y - (side_length / 2.0)))
    x2 = x1 + side_length
    y2 = y1 + side_length

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > image_width:
        x1 -= x2 - image_width
        x2 = image_width
    if y2 > image_height:
        y1 -= y2 - image_height
        y2 = image_height

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_width, x2)
    y2 = min(image_height, y2)
    return x1, y1, x2, y2


def _largest_connected_component(cv2_module: object, np_module: object, mask: object) -> object:
    label_count, labels, stats, _ = cv2_module.connectedComponentsWithStats(mask, 8)
    if label_count <= 1:
        return mask

    largest_label = 1 + int(np_module.argmax(stats[1:, cv2_module.CC_STAT_AREA]))
    component_mask = np_module.zeros_like(mask)
    component_mask[labels == largest_label] = 255
    return component_mask


def _rembg_session() -> object:
    global _REMBG_SESSION
    if _REMBG_SESSION is not None:
        return _REMBG_SESSION

    project_cache_dir = Path(__file__).resolve().parent.parent / ".cache"
    cache_dir = Path(
        os.environ.get(
            "U2NET_HOME",
            str(project_cache_dir / "u2net"),
        )
    )
    temp_dir = Path(
        os.environ.get(
            "AVATAR_TMPDIR",
            str(project_cache_dir / "tmp"),
        )
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("U2NET_HOME", str(cache_dir))
    os.environ.setdefault("TMPDIR", str(temp_dir))
    os.environ.setdefault("TEMP", str(temp_dir))
    os.environ.setdefault("TMP", str(temp_dir))

    from rembg import new_session

    _REMBG_SESSION = new_session(
        REMBG_MODEL_NAME,
        providers=["CPUExecutionProvider"],
    )
    return _REMBG_SESSION


def _extract_subject_mask_with_rembg(
    cv2_module: object,
    np_module: object,
    roi_image: object,
) -> object:
    encode_ok, encoded_image = cv2_module.imencode(".png", roi_image)
    if not encode_ok:
        raise RuntimeError("Could not encode ROI image for background removal.")

    from rembg import remove

    session = _rembg_session()
    result_image = remove(encoded_image.tobytes(), session=session)
    decoded_image = cv2_module.imdecode(
        np_module.frombuffer(result_image, dtype=np_module.uint8),
        cv2_module.IMREAD_UNCHANGED,
    )
    if decoded_image is None or len(decoded_image.shape) != 3 or decoded_image.shape[2] < 4:
        raise RuntimeError("Background removal did not return an alpha channel.")

    return decoded_image[:, :, 3]


def _component_for_face(
    cv2_module: object,
    np_module: object,
    mask: object,
    face_box: tuple[int, int, int, int],
) -> object:
    label_count, labels, stats, _ = cv2_module.connectedComponentsWithStats(mask, 8)
    if label_count <= 1:
        return mask

    mask_height, mask_width = mask.shape[:2]
    face_x, face_y, face_width, face_height = face_box
    center_x = max(0, min(mask_width - 1, int(round(face_x + (face_width * 0.50)))))
    center_y = max(0, min(mask_height - 1, int(round(face_y + (face_height * 0.56)))))
    center_label = int(labels[center_y, center_x])
    if center_label > 0:
        component_mask = np_module.zeros_like(mask)
        component_mask[labels == center_label] = 255
        return component_mask

    search_x1, search_y1, search_x2, search_y2 = _expand_bounds(
        mask.shape[:2],
        face_x,
        face_y,
        face_x + face_width,
        face_y + face_height,
        pad_x=int(round(face_width * 0.18)),
        pad_y=int(round(face_height * 0.18)),
    )
    search_region = labels[search_y1:search_y2, search_x1:search_x2]
    foreground_labels = [
        int(label)
        for label in np_module.unique(search_region)
        if int(label) > 0
    ]
    if foreground_labels:
        best_label = max(
            foreground_labels,
            key=lambda label: int(np_module.count_nonzero(search_region == label)),
        )
        component_mask = np_module.zeros_like(mask)
        component_mask[labels == best_label] = 255
        return component_mask

    return _largest_connected_component(cv2_module, np_module, mask)


def _build_face_roi_bounds(
    image_shape: tuple[int, int],
    face_box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    image_height, image_width = image_shape
    face_x, face_y, face_width, face_height = face_box
    side_padding = int(round(face_width * 0.46))
    top_padding = int(round(face_height * 0.62))
    bottom_padding = int(round(face_height * 0.40))
    return (
        max(0, face_x - side_padding),
        max(0, face_y - top_padding),
        min(image_width, face_x + face_width + side_padding),
        min(image_height, face_y + face_height + bottom_padding),
    )


def _local_face_box(
    face_box: tuple[int, int, int, int],
    roi_bounds: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    face_x, face_y, face_width, face_height = face_box
    roi_x1, roi_y1, _, _ = roi_bounds
    return face_x - roi_x1, face_y - roi_y1, face_width, face_height


def _ellipse_geometry(
    face_box: tuple[int, int, int, int],
    *,
    padding_ratio: float,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    face_x, face_y, face_width, face_height = face_box
    center = (
        int(round(face_x + (face_width * 0.50))),
        int(round(face_y + (face_height * 0.46))),
    )
    outer_axes = (
        max(1, int(round(face_width * (0.62 + (padding_ratio * 0.12))))),
        max(1, int(round(face_height * (0.82 + (padding_ratio * 0.18))))),
    )
    inner_axes = (
        max(1, int(round(face_width * 0.32))),
        max(1, int(round(face_height * 0.44))),
    )
    return center, outer_axes, inner_axes


def _face_limit_mask(
    cv2_module: object,
    np_module: object,
    mask_shape: tuple[int, int],
    face_box: tuple[int, int, int, int],
    *,
    padding_ratio: float,
) -> object:
    face_x, face_y, face_width, face_height = face_box
    y_coords, x_coords = np_module.indices(mask_shape, dtype=np_module.float32)

    def soft_ellipse(
        center_x: float,
        center_y: float,
        axis_x: float,
        axis_y: float,
        feather: float,
    ) -> object:
        normalized_x = (x_coords - center_x) / max(axis_x, 1.0)
        normalized_y = (y_coords - center_y) / max(axis_y, 1.0)
        radial_distance = np_module.sqrt((normalized_x ** 2) + (normalized_y ** 2))
        return 1.0 - np_module.clip((radial_distance - 1.0) / feather, 0.0, 1.0)

    crown_mask = soft_ellipse(
        face_x + (face_width * 0.50),
        face_y + (face_height * 0.40),
        face_width * (0.54 + (padding_ratio * 0.04)),
        face_height * (0.74 + (padding_ratio * 0.07)),
        0.22,
    )
    jaw_mask = soft_ellipse(
        face_x + (face_width * 0.50),
        face_y + (face_height * 0.80),
        face_width * (0.35 + (padding_ratio * 0.04)),
        face_height * (0.29 + (padding_ratio * 0.05)),
        0.20,
    )
    chin_mask = soft_ellipse(
        face_x + (face_width * 0.50),
        face_y + (face_height * 1.02),
        face_width * (0.22 + (padding_ratio * 0.03)),
        face_height * (0.17 + (padding_ratio * 0.04)),
        0.18,
    )
    limit_mask = np_module.maximum.reduce((crown_mask, jaw_mask, chin_mask))
    return np_module.clip(limit_mask * 255.0, 0.0, 255.0).astype(np_module.uint8)


def _reference_portrait_mask(
    np_module: object,
    mask_shape: tuple[int, int],
    face_box: tuple[int, int, int, int],
    *,
    padding_ratio: float,
    width_scale: float = 1.0,
) -> object:
    _, mask_width = mask_shape
    face_x, face_y, face_width, face_height = face_box
    row_positions = (
        np_module.arange(mask_shape[0], dtype=np_module.float32) - float(face_y)
    ) / max(float(face_height), 1.0)
    vertical_stops = np_module.array(
        [-0.14, -0.04, 0.20, 0.60, 0.92, 1.08],
        dtype=np_module.float32,
    )
    vertical_values = np_module.array(
        [0.0, 0.55, 1.0, 1.0, 0.58, 0.0],
        dtype=np_module.float32,
    )
    half_width_values = (
        np_module.array(
            [0.00, 0.24, 0.34, 0.30, 0.18, 0.00],
            dtype=np_module.float32,
        )
        + (padding_ratio * 0.04)
    ) * float(face_width) * max(width_scale, 0.1)
    vertical_alpha = np_module.interp(
        row_positions,
        vertical_stops,
        vertical_values,
        left=0.0,
        right=0.0,
    ).astype(np_module.float32)
    half_widths = np_module.interp(
        row_positions,
        vertical_stops,
        half_width_values,
        left=0.0,
        right=0.0,
    ).astype(np_module.float32)

    x_coords = np_module.arange(mask_width, dtype=np_module.float32).reshape(1, mask_width)
    center_x = float(face_x) + (float(face_width) * 0.50)
    width_map = half_widths.reshape(-1, 1)
    feather_map = np_module.maximum(float(face_width) * 0.09, width_map * 0.26)
    horizontal_alpha = 1.0 - np_module.clip(
        (np_module.abs(x_coords - center_x) - width_map) / np_module.maximum(feather_map, 1.0),
        0.0,
        1.0,
    )
    portrait_mask = horizontal_alpha * vertical_alpha.reshape(-1, 1)
    return np_module.clip(portrait_mask * 255.0, 0.0, 255.0).astype(np_module.uint8)


def _subject_prior_mask(
    cv2_module: object,
    np_module: object,
    image_shape: tuple[int, int],
    face_box: tuple[int, int, int, int],
    *,
    padding_ratio: float,
) -> object:
    image_height, image_width = image_shape
    face_x, face_y, face_width, face_height = face_box
    prior_mask = np_module.full(
        (image_height, image_width),
        cv2_module.GC_PR_BGD,
        dtype=np_module.uint8,
    )

    border_width = max(4, int(round(min(image_height, image_width) * 0.03)))
    prior_mask[:border_width, :] = cv2_module.GC_BGD
    prior_mask[:, :border_width] = cv2_module.GC_BGD
    prior_mask[-border_width:, :] = cv2_module.GC_BGD
    prior_mask[:, -border_width:] = cv2_module.GC_BGD

    focus_x1, focus_y1, focus_x2, focus_y2 = _expand_bounds(
        image_shape,
        face_x,
        face_y,
        face_x + face_width,
        face_y + face_height,
        pad_x=int(round(face_width * 0.65)),
        pad_y=int(round(face_height * 0.95)),
    )
    prior_mask[focus_y1:focus_y2, focus_x1:focus_x2] = cv2_module.GC_PR_FGD

    center = (
        int(round(face_x + (face_width * 0.50))),
        int(round(face_y + (face_height * 0.54))),
    )
    outer_axes = (
        max(1, int(round(face_width * (0.94 + (padding_ratio * 0.08))))),
        max(1, int(round(face_height * (1.28 + (padding_ratio * 0.12))))),
    )
    inner_axes = (
        max(1, int(round(face_width * 0.46))),
        max(1, int(round(face_height * 0.62))),
    )
    cv2_module.ellipse(
        prior_mask,
        center,
        outer_axes,
        0,
        0,
        360,
        cv2_module.GC_PR_FGD,
        -1,
    )
    cv2_module.ellipse(
        prior_mask,
        center,
        inner_axes,
        0,
        0,
        360,
        cv2_module.GC_FGD,
        -1,
    )

    core_x1 = max(0, face_x + int(round(face_width * 0.18)))
    core_y1 = max(0, face_y + int(round(face_height * 0.12)))
    core_x2 = min(image_width, face_x + face_width - int(round(face_width * 0.18)))
    core_y2 = min(image_height, face_y + face_height - int(round(face_height * 0.08)))
    prior_mask[core_y1:core_y2, core_x1:core_x2] = cv2_module.GC_FGD

    shoulder_cutoff = min(image_height, int(round(face_y + (face_height * 1.70))))
    prior_mask[shoulder_cutoff:, :] = np_module.minimum(
        prior_mask[shoulder_cutoff:, :],
        cv2_module.GC_PR_BGD,
    )
    return prior_mask


def _face_prior_mask(
    cv2_module: object,
    np_module: object,
    roi_shape: tuple[int, int],
    face_box: tuple[int, int, int, int],
    *,
    padding_ratio: float,
) -> object:
    roi_height, roi_width = roi_shape
    face_x, face_y, face_width, face_height = face_box
    center, outer_axes, inner_axes = _ellipse_geometry(
        face_box,
        padding_ratio=padding_ratio,
    )

    prior_mask = np_module.full((roi_height, roi_width), cv2_module.GC_BGD, dtype=np_module.uint8)
    region_x1, region_y1, region_x2, region_y2 = _expand_bounds(
        roi_shape,
        face_x,
        face_y,
        face_x + face_width,
        face_y + face_height,
        pad_x=int(round(face_width * 0.16)),
        pad_y=int(round(face_height * 0.10)),
    )
    prior_mask[region_y1:region_y2, region_x1:region_x2] = cv2_module.GC_PR_BGD
    cv2_module.ellipse(
        prior_mask,
        center,
        outer_axes,
        0,
        0,
        360,
        cv2_module.GC_PR_FGD,
        -1,
    )
    core_x1 = max(0, face_x + int(round(face_width * 0.20)))
    core_y1 = max(0, face_y + int(round(face_height * 0.18)))
    core_x2 = min(roi_width, face_x + face_width - int(round(face_width * 0.20)))
    core_y2 = min(roi_height, face_y + face_height - int(round(face_height * 0.10)))
    if core_x2 <= core_x1:
        core_x1, core_x2 = face_x, min(roi_width, face_x + face_width)
    if core_y2 <= core_y1:
        core_y1, core_y2 = face_y, min(roi_height, face_y + face_height)
    prior_mask[core_y1:core_y2, core_x1:core_x2] = cv2_module.GC_FGD
    cv2_module.ellipse(
        prior_mask,
        center,
        inner_axes,
        0,
        0,
        360,
        cv2_module.GC_PR_FGD,
        -1,
    )

    border_width = max(3, int(round(min(roi_height, roi_width) * 0.02)))
    prior_mask[:border_width, :] = cv2_module.GC_BGD
    prior_mask[:, :border_width] = cv2_module.GC_BGD
    prior_mask[-border_width:, :] = cv2_module.GC_BGD
    prior_mask[:, -border_width:] = cv2_module.GC_BGD

    shoulder_cutoff = min(roi_height, int(round(face_y + (face_height * 1.35))))
    prior_mask[shoulder_cutoff:, :] = np_module.minimum(
        prior_mask[shoulder_cutoff:, :],
        cv2_module.GC_PR_BGD,
    )
    return prior_mask


def _scale_face_box(
    face_box: tuple[int, int, int, int],
    scale: float,
) -> tuple[int, int, int, int]:
    face_x, face_y, face_width, face_height = face_box
    return (
        int(round(face_x * scale)),
        int(round(face_y * scale)),
        max(1, int(round(face_width * scale))),
        max(1, int(round(face_height * scale))),
    )


def _resize_for_segmentation(
    cv2_module: object,
    image: object,
    face_box: tuple[int, int, int, int],
) -> tuple[object, tuple[int, int, int, int], float]:
    height, width = image.shape[:2]
    longest_edge = max(height, width)
    if longest_edge <= DEFAULT_SEGMENTATION_MAX_EDGE:
        return image, face_box, 1.0

    scale = DEFAULT_SEGMENTATION_MAX_EDGE / float(longest_edge)
    resized_image = cv2_module.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2_module.INTER_AREA,
    )
    return resized_image, _scale_face_box(face_box, scale), scale


def _grabcut_rect(
    image_shape: tuple[int, int],
    face_box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    image_height, image_width = image_shape
    face_x, face_y, face_width, face_height = face_box
    rect_x1 = max(0, face_x - int(round(face_width * 0.12)))
    rect_y1 = max(0, face_y - int(round(face_height * 0.24)))
    rect_x2 = min(image_width, face_x + face_width + int(round(face_width * 0.12)))
    rect_y2 = min(image_height, face_y + face_height + int(round(face_height * 0.40)))
    return (
        rect_x1,
        rect_y1,
        max(1, rect_x2 - rect_x1),
        max(1, rect_y2 - rect_y1),
    )


def _trim_lower_face_region(
    np_module: object,
    face_mask: object,
    face_box: tuple[int, int, int, int],
) -> object:
    height, width = face_mask.shape
    face_x, face_y, face_width, face_height = face_box
    refined_mask = face_mask.copy()
    jaw_start = max(0, min(height, int(round(face_y + (face_height * 0.80)))))
    chin_end = max(jaw_start + 1, min(height, int(round(face_y + (face_height * 1.10)))))
    center_x = face_x + (face_width / 2.0)
    for row in range(jaw_start, chin_end):
        progress = min(1.0, max(0.0, (row - jaw_start) / max(chin_end - jaw_start, 1)))
        half_width = face_width * (0.36 - (0.14 * progress))
        left = max(0, min(width, int(round(center_x - half_width))))
        right = max(left, min(width, int(round(center_x + half_width))))
        refined_mask[row, :left] = 0
        refined_mask[row, right:] = 0
    refined_mask[chin_end:, :] = 0
    return refined_mask


def _retain_face_row_segments(
    np_module: object,
    face_mask: object,
    face_box: tuple[int, int, int, int],
) -> object:
    height, width = face_mask.shape
    face_x, _, face_width, _ = face_box
    anchor_x1 = max(0, min(width, int(round(face_x + (face_width * 0.18)))))
    anchor_x2 = max(anchor_x1 + 1, min(width, int(round(face_x + (face_width * 0.82)))))
    face_center_x = face_x + (face_width / 2.0)
    refined_mask = np_module.zeros_like(face_mask)

    for row in range(height):
        active_columns = np_module.flatnonzero(face_mask[row] > 0)
        if active_columns.size == 0:
            continue

        split_points = np_module.where(np_module.diff(active_columns) > 1)[0] + 1
        runs = np_module.split(active_columns, split_points)
        best_run: object | None = None
        best_score: tuple[int, int, float, int] | None = None
        for run in runs:
            run_start = int(run[0])
            run_end = int(run[-1]) + 1
            overlap = max(0, min(run_end, anchor_x2) - max(run_start, anchor_x1))
            run_center = run_start + ((run_end - run_start) / 2.0)
            score = (
                1 if overlap > 0 else 0,
                overlap,
                -abs(run_center - face_center_x),
                len(run),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_run = run

        if best_run is not None:
            refined_mask[row, int(best_run[0]) : int(best_run[-1]) + 1] = 255

    return refined_mask


def _limit_face_row_widths(
    np_module: object,
    face_mask: object,
    face_box: tuple[int, int, int, int],
) -> object:
    height, width = face_mask.shape
    face_x, face_y, face_width, face_height = face_box
    face_center_x = face_x + (face_width / 2.0)
    start_row = max(0, int(round(face_y - (face_height * 0.06))))
    end_row = min(height, int(round(face_y + (face_height * 1.02))))
    refined_mask = face_mask.copy()

    for row in range(start_row, end_row):
        vertical_position = (row - face_y) / max(face_height, 1)
        if vertical_position < 0.12:
            half_width = face_width * 0.56
        elif vertical_position < 0.50:
            progress = (vertical_position - 0.12) / 0.38
            half_width = face_width * (0.56 - (0.06 * progress))
        elif vertical_position < 0.82:
            progress = (vertical_position - 0.50) / 0.32
            half_width = face_width * (0.50 - (0.08 * progress))
        else:
            progress = min(1.0, max(0.0, (vertical_position - 0.82) / 0.20))
            half_width = face_width * (0.42 - (0.16 * progress))

        left = max(0, min(width, int(round(face_center_x - half_width))))
        right = max(left, min(width, int(round(face_center_x + half_width))))
        refined_mask[row, :left] = 0
        refined_mask[row, right:] = 0

    return refined_mask


def _extract_subject_mask(
    cv2_module: object,
    np_module: object,
    roi_image: object,
    face_box: tuple[int, int, int, int],
    *,
    padding_ratio: float,
) -> object:
    del padding_ratio
    subject_mask = _extract_subject_mask_with_rembg(
        cv2_module,
        np_module,
        roi_image,
    )

    binary_subject_mask = np_module.where(
        subject_mask >= DEFAULT_MASK_BOUNDS_THRESHOLD,
        255,
        0,
    ).astype(np_module.uint8)
    close_kernel_size = max(
        5,
        int(round(min(face_box[2], face_box[3]) * 0.04)),
    )
    if close_kernel_size % 2 == 0:
        close_kernel_size += 1
    close_kernel = cv2_module.getStructuringElement(
        cv2_module.MORPH_ELLIPSE,
        (close_kernel_size, close_kernel_size),
    )
    binary_subject_mask = cv2_module.morphologyEx(
        binary_subject_mask,
        cv2_module.MORPH_CLOSE,
        close_kernel,
    )
    open_kernel_size = max(
        3,
        int(round(min(face_box[2], face_box[3]) * 0.018)),
    )
    if open_kernel_size % 2 == 0:
        open_kernel_size += 1
    open_kernel = cv2_module.getStructuringElement(
        cv2_module.MORPH_ELLIPSE,
        (open_kernel_size, open_kernel_size),
    )
    binary_subject_mask = cv2_module.morphologyEx(
        binary_subject_mask,
        cv2_module.MORPH_OPEN,
        open_kernel,
    )
    binary_subject_mask = _component_for_face(
        cv2_module,
        np_module,
        binary_subject_mask,
        face_box,
    )
    if int(binary_subject_mask.max()) == 0:
        raise RuntimeError("Background removal did not keep any face pixels.")
    cleaned_subject_mask = np_module.where(binary_subject_mask > 0, subject_mask, 0).astype(
        np_module.uint8
    )
    return cleaned_subject_mask


def _extract_face_mask(
    cv2_module: object,
    np_module: object,
    subject_mask: object,
    face_box: tuple[int, int, int, int],
    *,
    padding_ratio: float,
) -> object:
    portrait_mask = _reference_portrait_mask(
        np_module,
        subject_mask.shape[:2],
        face_box,
        padding_ratio=padding_ratio,
    )
    subject_alpha = subject_mask.astype(np_module.float32) / 255.0
    subject_alpha = np_module.clip(
        (subject_alpha - 0.02) / 0.98,
        0.0,
        1.0,
    )
    support_mask = np_module.where(subject_mask >= DEFAULT_MASK_BOUNDS_THRESHOLD, 255, 0).astype(
        np_module.uint8
    )
    support_mask = _component_for_face(cv2_module, np_module, support_mask, face_box)
    support_alpha = cv2_module.GaussianBlur(
        support_mask.astype(np_module.float32) / 255.0,
        (0, 0),
        max(1.0, min(face_box[2], face_box[3]) * 0.015),
    )
    subject_alpha = np_module.maximum(subject_alpha, support_alpha * 0.85)
    portrait_alpha = portrait_mask.astype(np_module.float32) / 255.0
    face_mask = np_module.minimum(subject_alpha, portrait_alpha)
    blur_size = max(5, int(round(min(face_box[2], face_box[3]) * 0.040)))
    if blur_size % 2 == 0:
        blur_size += 1
    return cv2_module.GaussianBlur(
        np_module.clip(face_mask * 255.0, 0.0, 255.0).astype(np_module.uint8),
        (blur_size, blur_size),
        0,
    )


def _mask_bounds(np_module: object, mask: object) -> tuple[int, int, int, int]:
    mask_points = np_module.column_stack(np_module.where(mask >= DEFAULT_MASK_BOUNDS_THRESHOLD))
    if mask_points.size == 0:
        raise RuntimeError("The face isolation mask is empty.")

    y_coordinates = mask_points[:, 0]
    x_coordinates = mask_points[:, 1]
    return (
        int(x_coordinates.min()),
        int(y_coordinates.min()),
        int(x_coordinates.max()) + 1,
        int(y_coordinates.max()) + 1,
    )


def _head_mask_bounds(
    np_module: object,
    mask: object,
    face_box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    mask_height, mask_width = mask.shape[:2]
    face_x, face_y, face_width, face_height = face_box
    focus_x1 = max(0, int(round(face_x - (face_width * 0.42))))
    focus_y1 = max(0, int(round(face_y - (face_height * 0.52))))
    focus_x2 = min(mask_width, int(round(face_x + face_width + (face_width * 0.42))))
    focus_y2 = min(mask_height, int(round(face_y + face_height + (face_height * 0.08))))
    if focus_x2 <= focus_x1 or focus_y2 <= focus_y1:
        return _mask_bounds(np_module, mask)

    focused_mask = mask[focus_y1:focus_y2, focus_x1:focus_x2]
    focused_points = np_module.column_stack(
        np_module.where(focused_mask >= DEFAULT_MASK_BOUNDS_THRESHOLD)
    )
    if focused_points.size == 0:
        return _mask_bounds(np_module, mask)

    y_coordinates = focused_points[:, 0] + focus_y1
    x_coordinates = focused_points[:, 1] + focus_x1
    return (
        int(x_coordinates.min()),
        int(y_coordinates.min()),
        int(x_coordinates.max()) + 1,
        int(y_coordinates.max()) + 1,
    )


def _build_output_bounds(
    image_shape: tuple[int, int],
    mask_bounds: tuple[int, int, int, int],
    face_box: tuple[int, int, int, int],
    *,
    padding_ratio: float,
    square: bool,
) -> tuple[int, int, int, int]:
    image_height, image_width = image_shape
    mask_x1, mask_y1, mask_x2, mask_y2 = mask_bounds
    mask_width = mask_x2 - mask_x1
    mask_height = mask_y2 - mask_y1
    pad_x = int(round(max(mask_width, face_box[2]) * padding_ratio))
    pad_y = int(round(max(mask_height, face_box[3]) * padding_ratio))
    crop_x1, crop_y1, crop_x2, crop_y2 = _expand_bounds(
        image_shape,
        mask_x1,
        mask_y1,
        mask_x2,
        mask_y2,
        pad_x=pad_x,
        pad_y=pad_y,
    )

    if not square:
        return crop_x1, crop_y1, crop_x2, crop_y2

    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1
    side_length = max(crop_width, crop_height)
    center_x = crop_x1 + (crop_width / 2.0)
    center_y = crop_y1 + (crop_height / 2.0) - (face_box[3] * 0.12)
    return _clamp_square_crop(
        image_width=image_width,
        image_height=image_height,
        center_x=center_x,
        center_y=center_y,
        side_length=side_length,
    )


def _composite_face_crop(
    cv2_module: object,
    np_module: object,
    image: object,
    face_mask: object,
    output_bounds: tuple[int, int, int, int],
    *,
    transparent_background: bool,
) -> object:
    x1, y1, x2, y2 = output_bounds
    cropped_image = image[y1:y2, x1:x2]
    cropped_mask = face_mask[y1:y2, x1:x2]
    alpha = cropped_mask.astype(np_module.float32) / 255.0
    alpha_floor = DEFAULT_ALPHA_TRIM / 255.0
    alpha = np_module.clip((alpha - alpha_floor) / max(1.0 - alpha_floor, 1e-6), 0.0, 1.0)
    output_alpha = np_module.clip(alpha * 255.0, 0.0, 255.0).astype(np_module.uint8)
    color_crop = (
        cropped_image.astype(np_module.float32) * alpha[:, :, None]
    ).astype(np_module.uint8)

    if not transparent_background:
        return color_crop

    output_image = cv2_module.cvtColor(cropped_image, cv2_module.COLOR_BGR2BGRA)
    output_image[:, :, 3] = output_alpha
    output_image[output_alpha == 0, :3] = 0
    return output_image


def crop_face_image(
    source_image_path: Union[Path, str],
    output_path: Union[Path, str],
    *,
    padding_ratio: float = DEFAULT_PADDING_RATIO,
    square: bool = True,
    transparent_background: bool = False,
) -> Path:
    if padding_ratio < 0:
        raise ValueError("padding_ratio must be 0 or greater.")

    source_image = Path(source_image_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()

    _validate_source_image(source_image)
    _validate_runtime_dependencies()

    if transparent_background and output_file.suffix.lower() != ".png":
        raise ValueError("Transparent output requires a .png destination.")

    import cv2
    import numpy as np

    image = _load_image(cv2, source_image)
    face_box = _detect_primary_face(cv2, image)
    roi_bounds = _build_face_roi_bounds(image.shape[:2], face_box)
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_bounds
    roi_image = image[roi_y1:roi_y2, roi_x1:roi_x2]
    local_face_box = _local_face_box(face_box, roi_bounds)
    subject_mask = _extract_subject_mask(
        cv2,
        np,
        roi_image,
        local_face_box,
        padding_ratio=padding_ratio,
    )
    # Keep the run_face output focused on the facial portrait rather than the
    # entire segmented head-and-shoulders silhouette.
    local_face_mask = _extract_face_mask(
        cv2,
        np,
        subject_mask,
        local_face_box,
        padding_ratio=padding_ratio,
    )
    local_mask_bounds = _head_mask_bounds(np, local_face_mask, local_face_box)
    local_output_bounds = _build_output_bounds(
        roi_image.shape[:2],
        local_mask_bounds,
        local_face_box,
        padding_ratio=padding_ratio,
        square=square,
    )
    isolated_face = _composite_face_crop(
        cv2,
        np,
        roi_image,
        local_face_mask,
        local_output_bounds,
        transparent_background=transparent_background,
    )
    if isolated_face.size == 0:
        raise RuntimeError("Face crop produced an empty output image.")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_file), isolated_face):
        raise RuntimeError(f"Could not write cropped face image to {output_file}")

    return output_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect the main face in a photo and save an isolated face-only crop."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the source photo.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path where the cropped face image should be saved.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=DEFAULT_PADDING_RATIO,
        help="Extra border around the isolated face as a fraction of the detected face size.",
    )
    parser.add_argument(
        "--no-square",
        action="store_true",
        help="Keep the output rectangular instead of forcing a square canvas.",
    )
    parser.add_argument(
        "--transparent-background",
        action="store_true",
        help="Write the non-face area as transparency instead of black. Requires PNG output.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = crop_face_image(
        source_image_path=args.input,
        output_path=args.output,
        padding_ratio=args.padding,
        square=not args.no_square,
        transparent_background=args.transparent_background,
    )
    print(f"Saved cropped face image to {output_path}")


if __name__ == "__main__":
    main()
