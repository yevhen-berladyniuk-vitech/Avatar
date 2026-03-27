from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf
import torch

DEFAULT_REPO_ID = "hexgrad/Kokoro-82M"


def _install_weight_norm_compat() -> None:
    parametrizations = getattr(torch.nn.utils, "parametrizations", None)
    if parametrizations is None or not hasattr(parametrizations, "weight_norm"):
        return

    if getattr(torch.nn.utils, "_avatar_weight_norm_patched", False):
        return

    def _compat_weight_norm(module: torch.nn.Module, name: str = "weight", dim: int = 0):
        return parametrizations.weight_norm(module, name=name, dim=dim)

    torch.nn.utils.weight_norm = _compat_weight_norm
    torch.nn.utils._avatar_weight_norm_patched = True


_install_weight_norm_compat()

from kokoro import KPipeline

DEFAULT_SAMPLE_RATE = 24000
DEFAULT_LANG_CODE = "a"
DEFAULT_VOICE = "af_heart"


def _chunk_to_numpy(audio_chunk: object) -> np.ndarray:
    if hasattr(audio_chunk, "detach"):
        audio_chunk = audio_chunk.detach().cpu().numpy()

    return np.asarray(audio_chunk, dtype=np.float32).reshape(-1)


def generate_speech(
    text: str,
    output_path: Union[str, Path],
    voice: str = DEFAULT_VOICE,
    lang_code: str = DEFAULT_LANG_CODE,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Path:
    if not text.strip():
        raise ValueError("Text for TTS must not be empty.")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    pipeline = KPipeline(lang_code=lang_code, repo_id=DEFAULT_REPO_ID)
    audio_segments = []

    for _, _, audio_chunk in pipeline(text, voice=voice):
        segment = _chunk_to_numpy(audio_chunk)
        if segment.size:
            audio_segments.append(segment)

    if not audio_segments:
        raise RuntimeError("Kokoro did not return any audio segments.")

    combined_audio = np.concatenate(audio_segments)
    sf.write(output_file, combined_audio, sample_rate)
    return output_file
