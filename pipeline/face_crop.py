from __future__ import annotations

import argparse
import importlib.util
import math
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
}
CASCADE_FILENAMES: Final = (
    "haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_alt.xml",
    "haarcascade_frontalface_alt2.xml",
    "haarcascade_profileface.xml",
)
DEFAULT_PADDING_RATIO: Final = 0.10
DEFAULT_MAX_DETECTION_EDGE: Final = 1280
DEFAULT_MIN_FACE_SIZE: Final = 24


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


def _build_face_roi_bounds(
    image_shape: tuple[int, int],
    face_box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    face_x, face_y, face_width, face_height = face_box
    return _expand_bounds(
        image_shape,
        face_x,
        face_y,
        face_x + face_width,
        face_y + face_height,
        pad_x=int(round(face_width * 0.28)),
        pad_y=int(round(face_height * 0.18)),
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
        int(round(face_y + (face_height * 0.50))),
    )
    outer_axes = (
        max(1, int(round(face_width * (0.35 + (padding_ratio * 0.25))))),
        max(1, int(round(face_height * (0.44 + (padding_ratio * 0.35))))),
    )
    inner_axes = (
        max(1, int(round(outer_axes[0] * 0.76))),
        max(1, int(round(outer_axes[1] * 0.78))),
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
    limit_mask = np_module.zeros(mask_shape, dtype=np_module.uint8)
    center, outer_axes, _ = _ellipse_geometry(face_box, padding_ratio=padding_ratio)
    cv2_module.ellipse(limit_mask, center, outer_axes, 0, 0, 360, 255, -1)
    return limit_mask


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

    border_width = max(3, int(round(min(roi_height, roi_width) * 0.02)))
    prior_mask[:border_width, :] = cv2_module.GC_BGD
    prior_mask[:, :border_width] = cv2_module.GC_BGD
    prior_mask[-border_width:, :] = cv2_module.GC_BGD
    prior_mask[:, -border_width:] = cv2_module.GC_BGD

    neck_cutoff = min(roi_height, int(round(face_y + (face_height * 1.00))))
    prior_mask[neck_cutoff:, :] = cv2_module.GC_BGD
    return prior_mask


def _extract_face_mask(
    cv2_module: object,
    np_module: object,
    roi_image: object,
    face_box: tuple[int, int, int, int],
    *,
    padding_ratio: float,
) -> object:
    face_limit_mask = _face_limit_mask(
        cv2_module,
        np_module,
        roi_image.shape[:2],
        face_box,
        padding_ratio=padding_ratio,
    )
    grabcut_mask = _face_prior_mask(
        cv2_module,
        np_module,
        roi_image.shape[:2],
        face_box,
        padding_ratio=padding_ratio,
    )

    background_model = np_module.zeros((1, 65), dtype=np_module.float64)
    foreground_model = np_module.zeros((1, 65), dtype=np_module.float64)
    try:
        cv2_module.grabCut(
            roi_image,
            grabcut_mask,
            None,
            background_model,
            foreground_model,
            5,
            cv2_module.GC_INIT_WITH_MASK,
        )
        face_mask = np_module.where(
            (grabcut_mask == cv2_module.GC_FGD)
            | (grabcut_mask == cv2_module.GC_PR_FGD),
            255,
            0,
        ).astype(np_module.uint8)
    except cv2_module.error:
        face_mask = face_limit_mask.copy()

    face_mask = cv2_module.bitwise_and(face_mask, face_limit_mask)
    face_mask = _largest_connected_component(cv2_module, np_module, face_mask)

    close_kernel_size = max(3, int(round(min(face_box[2], face_box[3]) * 0.06)))
    if close_kernel_size % 2 == 0:
        close_kernel_size += 1
    close_kernel = cv2_module.getStructuringElement(
        cv2_module.MORPH_ELLIPSE,
        (close_kernel_size, close_kernel_size),
    )
    face_mask = cv2_module.morphologyEx(face_mask, cv2_module.MORPH_CLOSE, close_kernel)
    face_mask = _largest_connected_component(cv2_module, np_module, face_mask)

    blur_size = max(5, int(round(min(face_box[2], face_box[3]) * 0.08)))
    if blur_size % 2 == 0:
        blur_size += 1
    return cv2_module.GaussianBlur(face_mask, (blur_size, blur_size), 0)


def _mask_bounds(np_module: object, mask: object) -> tuple[int, int, int, int]:
    mask_points = np_module.column_stack(np_module.where(mask > 0))
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
    center_y = crop_y1 + (crop_height / 2.0)
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
    color_crop = (
        cropped_image.astype(np_module.float32) * alpha[:, :, None]
    ).astype(np_module.uint8)

    if not transparent_background:
        return color_crop

    output_image = cv2_module.cvtColor(color_crop, cv2_module.COLOR_BGR2BGRA)
    output_image[:, :, 3] = cropped_mask
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
    local_face_mask = _extract_face_mask(
        cv2,
        np,
        roi_image,
        local_face_box,
        padding_ratio=padding_ratio,
    )
    local_mask_bounds = _mask_bounds(np, local_face_mask)
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
