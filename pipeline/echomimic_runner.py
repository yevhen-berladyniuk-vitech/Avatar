from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file as load_safetensors_file

try:
    IMAGE_RESAMPLING = Image.Resampling
except AttributeError:
    IMAGE_RESAMPLING = Image


def _log(message: str) -> None:
    print(message, flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--echomimic-dir", required=True)
    parser.add_argument("--variant", choices=["standard", "accelerated"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-L", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--facemusk_dilation_ratio", type=float, default=0.1)
    parser.add_argument("--facecrop_dilation_ratio", type=float, default=0.5)
    parser.add_argument("--context_frames", type=int, default=12)
    parser.add_argument("--context_overlap", type=int, default=3)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _resolve_device(requested_device: str) -> str:
    normalized_device = requested_device.strip().lower()
    if normalized_device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        return "cpu"
    return requested_device


def _should_use_face_detection(device: torch.device) -> bool:
    return device.type not in {"cpu", "mps"} and sys.platform != "darwin"


def _crop_and_pad(image: np.ndarray, rect: list[int]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x0, y0, x1, y1 = rect
    height, width = image.shape[:2]

    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)

    rect_width = x1 - x0
    rect_height = y1 - y0
    side_length = min(rect_width, rect_height)

    center_x = (x0 + x1) // 2
    center_y = (y0 + y1) // 2

    new_x0 = max(0, center_x - side_length // 2)
    new_y0 = max(0, center_y - side_length // 2)
    new_x1 = min(width, new_x0 + side_length)
    new_y1 = min(height, new_y0 + side_length)

    if (new_x1 - new_x0) != (new_y1 - new_y0):
        side_length = min(new_x1 - new_x0, new_y1 - new_y0)
        new_x1 = new_x0 + side_length
        new_y1 = new_y0 + side_length

    cropped_image = image[new_y0:new_y1, new_x0:new_x1]
    return cropped_image, (new_x0, new_y0, new_x1, new_y1)


def _select_face(det_bboxes: np.ndarray | None, probs: np.ndarray | None) -> np.ndarray | None:
    if det_bboxes is None or probs is None:
        return None

    filtered_bboxes = [
        det_bboxes[index]
        for index in range(len(det_bboxes))
        if probs[index] > 0.8
    ]
    if not filtered_bboxes:
        return None

    filtered_bboxes.sort(
        key=lambda bbox: (bbox[3] - bbox[1]) * (bbox[2] - bbox[0]),
        reverse=True,
    )
    return filtered_bboxes[0]


def _resolve_executable(candidate: str) -> Path | None:
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
        "EchoMimic runner could not locate an ffmpeg executable. "
        "Install ffmpeg or set FFMPEG/FFMPEG_PATH."
    )


def _run_command(command: list[str]) -> None:
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        rendered_command = " ".join(command)
        combined_output = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(
            f"Command failed: {rendered_command}. stderr: {combined_output or '<empty>'}"
        ) from exc


def _load_rgb_image(image_path: str) -> np.ndarray:
    with Image.open(image_path) as image:
        return np.asarray(image.convert("RGB"))


def _resize_rgb_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        Image.fromarray(image).resize(size, resample=IMAGE_RESAMPLING.BICUBIC)
    )


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        Image.fromarray(mask).resize(size, resample=IMAGE_RESAMPLING.NEAREST)
    )


def _write_video(video: torch.Tensor | np.ndarray, output_path: Path, fps: int) -> Path:
    if isinstance(video, torch.Tensor):
        video_array = video.detach().cpu().numpy()
    else:
        video_array = np.asarray(video)

    if video_array.ndim != 5:
        raise RuntimeError(f"Unexpected EchoMimic output shape: {video_array.shape}")

    frames = np.transpose(video_array[0], (1, 2, 3, 0))
    frames = np.clip(frames * 255.0, 0, 255).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_executable = _find_ffmpeg_executable()
    with tempfile.TemporaryDirectory(prefix="echomimic_frames_") as frame_dir:
        frame_dir_path = Path(frame_dir)
        for index, frame in enumerate(frames):
            frame_path = frame_dir_path / f"frame_{index:06d}.png"
            Image.fromarray(frame).save(frame_path)

        command = [
            str(ffmpeg_executable),
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir_path / "frame_%06d.png"),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        _run_command(command)

    if not output_path.is_file():
        raise RuntimeError(f"EchoMimic runner did not produce {output_path}")
    return output_path


def _preferred_checkpoint_path(checkpoint_path: str) -> Path:
    resolved_checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    safetensors_path = resolved_checkpoint_path.with_suffix(".safetensors")
    if safetensors_path.is_file():
        return safetensors_path
    return resolved_checkpoint_path


def _load_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
    resolved_checkpoint_path = _preferred_checkpoint_path(checkpoint_path)
    if resolved_checkpoint_path.suffix.lower() == ".safetensors":
        return load_safetensors_file(str(resolved_checkpoint_path), device="cpu")

    load_kwargs = {"map_location": "cpu"}
    try:
        return torch.load(str(resolved_checkpoint_path), weights_only=True, **load_kwargs)
    except TypeError:
        return torch.load(str(resolved_checkpoint_path), **load_kwargs)


def _apply_state_dict(
    module: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> tuple[list[str], list[str]]:
    model_state = module.state_dict()
    loaded_keys: set[str] = set()
    skipped_keys: list[str] = []
    unexpected_keys: list[str] = []

    with torch.no_grad():
        for key, value in state_dict.items():
            target = model_state.get(key)
            if target is None:
                unexpected_keys.append(key)
                continue
            if target.shape != value.shape:
                skipped_keys.append(key)
                continue

            target.copy_(value.to(device=target.device, dtype=target.dtype))
            loaded_keys.add(key)

    missing_keys = [key for key in model_state.keys() if key not in loaded_keys]
    missing_keys.extend(skipped_keys)
    return missing_keys, unexpected_keys


def _patch_reference_attention_control(device: torch.device) -> None:
    from src.models import mutual_self_attention

    reference_control = mutual_self_attention.ReferenceAttentionControl
    if not hasattr(reference_control, "_avatar_original_register_reference_hooks"):
        reference_control._avatar_original_register_reference_hooks = (
            reference_control.register_reference_hooks
        )

    original_method = reference_control._avatar_original_register_reference_hooks

    def _patched_register_reference_hooks(self, *args, **kwargs):
        kwargs.setdefault("device", device)
        return original_method(self, *args, **kwargs)

    reference_control.register_reference_hooks = _patched_register_reference_hooks


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

    echomimic_dir = Path(args.echomimic_dir).expanduser().resolve()
    if str(echomimic_dir) not in sys.path:
        sys.path.insert(0, str(echomimic_dir))

    from src.models.face_locator import FaceLocator
    from src.models.unet_2d_condition import UNet2DConditionModel
    from src.models.unet_3d_echo import EchoUNet3DConditionModel
    from src.models.whisper.audio2feature import load_audio_model

    if args.variant == "accelerated":
        from src.pipelines.pipeline_echo_mimic_acc import Audio2VideoPipeline
    else:
        from src.pipelines.pipeline_echo_mimic import Audio2VideoPipeline

    config = OmegaConf.load(args.config)
    inference_config = OmegaConf.load(config.inference_config)
    device_name = _resolve_device(args.device)
    device = torch.device(device_name)
    weight_dtype = torch.float16 if device.type == "cuda" else torch.float32

    _log(f"EchoMimic runner variant: {args.variant}")
    _log(f"EchoMimic runner device: {device}")

    _patch_reference_attention_control(device)

    _log("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(config.pretrained_vae_path).to(
        device=device,
        dtype=weight_dtype,
    )
    _log("Loading reference UNet...")
    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(device=device, dtype=weight_dtype)
    _log("Loading reference checkpoint weights...")
    reference_state_dict = _load_state_dict(config.reference_unet_path)
    _log("Applying reference checkpoint weights...")
    reference_missing, reference_unexpected = _apply_state_dict(
        reference_unet,
        reference_state_dict,
    )
    _log(
        "Reference checkpoint applied "
        f"(missing={len(reference_missing)}, unexpected={len(reference_unexpected)})."
    )

    _log("Loading denoising UNet...")
    motion_module_path = str(_preferred_checkpoint_path(config.motion_module_path))
    if os.path.exists(config.motion_module_path):
        denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
            config.pretrained_base_model_path,
            motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=inference_config.unet_additional_kwargs,
        ).to(device=device, dtype=weight_dtype)
    else:
        denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
            config.pretrained_base_model_path,
            "",
            subfolder="unet",
            unet_additional_kwargs={
                "use_motion_module": False,
                "unet_use_temporal_attention": False,
                "cross_attention_dim": inference_config.unet_additional_kwargs.cross_attention_dim,
            },
        ).to(device=device, dtype=weight_dtype)
    _log("Loading denoising checkpoint weights...")
    denoising_state_dict = _load_state_dict(config.denoising_unet_path)
    _log("Applying denoising checkpoint weights...")
    denoising_missing, denoising_unexpected = _apply_state_dict(
        denoising_unet,
        denoising_state_dict,
    )
    _log(
        "Denoising checkpoint applied "
        f"(missing={len(denoising_missing)}, unexpected={len(denoising_unexpected)})."
    )

    _log("Loading face locator...")
    face_locator = FaceLocator(
        320,
        conditioning_channels=1,
        block_out_channels=(16, 32, 96, 256),
    ).to(device=device, dtype=weight_dtype)
    face_locator_missing, face_locator_unexpected = _apply_state_dict(
        face_locator,
        _load_state_dict(config.face_locator_path),
    )
    _log(
        "Face locator checkpoint applied "
        f"(missing={len(face_locator_missing)}, unexpected={len(face_locator_unexpected)})."
    )

    _log("Loading audio processor...")
    audio_processor = load_audio_model(
        model_path=config.audio_model_path,
        device="cpu",
    )
    use_face_detection = _should_use_face_detection(device)
    face_detector = None
    if use_face_detection:
        _log("Loading face detector...")
        from facenet_pytorch import MTCNN

        face_detector = MTCNN(
            image_size=320,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device="cpu",
        )
    else:
        _log("Skipping face detection and using a full-frame mask.")

    scheduler = DDIMScheduler(
        **OmegaConf.to_container(inference_config.noise_scheduler_kwargs)
    )
    _log("Creating audio-to-video pipeline...")
    pipe = Audio2VideoPipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        audio_guider=audio_processor,
        face_locator=face_locator,
        scheduler=scheduler,
    ).to(device=device, dtype=weight_dtype)

    for ref_image_path, audio_paths in config["test_cases"].items():
        for audio_path in audio_paths:
            if args.seed is not None and args.seed > -1:
                generator = torch.manual_seed(args.seed)
            else:
                generator = torch.manual_seed(random.randint(100, 1_000_000))

            _log(f"Preparing reference image: {ref_image_path}")
            face_img = _load_rgb_image(ref_image_path)

            face_mask = np.zeros((face_img.shape[0], face_img.shape[1]), dtype="uint8")
            selected_bbox = None
            if face_detector is not None:
                det_bboxes, probs = face_detector.detect(Image.fromarray(face_img))
                selected_bbox = _select_face(det_bboxes, probs)

            if selected_bbox is None:
                face_mask[:, :] = 255
            else:
                xyxy = np.round(selected_bbox[:4]).astype("int")
                row_begin, row_end, col_begin, col_end = (
                    xyxy[1],
                    xyxy[3],
                    xyxy[0],
                    xyxy[2],
                )
                row_pad = int((row_end - row_begin) * args.facemusk_dilation_ratio)
                col_pad = int((col_end - col_begin) * args.facemusk_dilation_ratio)
                face_mask[
                    row_begin - row_pad : row_end + row_pad,
                    col_begin - col_pad : col_end + col_pad,
                ] = 255

                row_crop_pad = int((row_end - row_begin) * args.facecrop_dilation_ratio)
                col_crop_pad = int((col_end - col_begin) * args.facecrop_dilation_ratio)
                crop_rect = [
                    max(0, col_begin - col_crop_pad),
                    max(0, row_begin - row_crop_pad),
                    min(col_end + col_crop_pad, face_img.shape[1]),
                    min(row_end + row_crop_pad, face_img.shape[0]),
                ]
                face_img, _ = _crop_and_pad(face_img, crop_rect)
                face_mask, _ = _crop_and_pad(face_mask, crop_rect)

            face_img = _resize_rgb_image(face_img, (args.W, args.H))
            face_mask = _resize_mask(face_mask, (args.W, args.H))

            ref_image_pil = Image.fromarray(face_img)
            face_mask_tensor = (
                torch.tensor(face_mask, dtype=weight_dtype, device=device)
                .unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(0)
                / 255.0
            )

            _log("Running diffusion...")
            video = pipe(
                ref_image_pil,
                audio_path,
                face_mask_tensor,
                args.W,
                args.H,
                args.L,
                args.steps,
                args.cfg,
                generator=generator,
                audio_sample_rate=args.sample_rate,
                context_frames=args.context_frames,
                fps=args.fps,
                context_overlap=args.context_overlap,
            ).videos

            output_path = Path(args.output).expanduser().resolve()
            _log(f"Encoding silent MP4 to {output_path}...")
            _write_video(video, output_path, args.fps)
            print(output_path)
            return

    raise RuntimeError("EchoMimic runner found no test cases in the generated config.")


if __name__ == "__main__":
    main()
