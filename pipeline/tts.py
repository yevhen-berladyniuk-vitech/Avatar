import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Optional, Union

DEFAULT_REPO_ID = "hexgrad/Kokoro-82M"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_LANG_CODE = "a"
DEFAULT_VOICE = "af_heart"


def _install_weight_norm_compat(torch_module: object) -> None:
    nn_utils = getattr(torch_module, "nn", None)
    if nn_utils is None:
        return

    utils = getattr(nn_utils, "utils", None)
    if utils is None:
        return

    parametrizations = getattr(utils, "parametrizations", None)
    if parametrizations is None or not hasattr(parametrizations, "weight_norm"):
        return

    if getattr(utils, "_avatar_weight_norm_patched", False):
        return

    def _compat_weight_norm(module: object, name: str = "weight", dim: int = 0):
        return parametrizations.weight_norm(module, name=name, dim=dim)

    utils.weight_norm = _compat_weight_norm
    utils._avatar_weight_norm_patched = True


def _chunk_to_numpy(audio_chunk: object) -> "np.ndarray":
    import numpy as np

    if hasattr(audio_chunk, "detach"):
        audio_chunk = audio_chunk.detach().cpu().numpy()

    return np.asarray(audio_chunk, dtype=np.float32).reshape(-1)


def _generate_speech_with_kokoro(
    text: str,
    output_file: Path,
    voice: Optional[str],
    lang_code: str,
    sample_rate: int,
) -> Path:
    import numpy as np
    import soundfile as sf
    import torch
    from kokoro import KPipeline

    _install_weight_norm_compat(torch)

    pipeline = KPipeline(lang_code=lang_code, repo_id=DEFAULT_REPO_ID)
    audio_segments = []
    selected_voice = voice or DEFAULT_VOICE

    for _, _, audio_chunk in pipeline(text, voice=selected_voice):
        segment = _chunk_to_numpy(audio_chunk)
        if segment.size:
            audio_segments.append(segment)

    if not audio_segments:
        raise RuntimeError("Kokoro did not return any audio segments.")

    combined_audio = np.concatenate(audio_segments)
    sf.write(output_file, combined_audio, sample_rate)
    _validate_wav_file(output_file)
    return output_file


def _generate_speech_with_macos_say(
    text: str,
    output_file: Path,
    voice: Optional[str],
    sample_rate: int,
) -> Path:
    say_executable = shutil.which("say")
    afconvert_executable = shutil.which("afconvert")

    if say_executable is None or afconvert_executable is None:
        raise RuntimeError(
            "macOS speech synthesis requires both 'say' and 'afconvert' to be available."
        )

    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as temp_audio_file:
        temp_aiff_path = Path(temp_audio_file.name)

    say_command = [say_executable]
    if voice:
        say_command.extend(["-v", voice])
    say_command.extend(["-o", str(temp_aiff_path), text])

    try:
        subprocess.run(say_command, check=True)
        subprocess.run(
            [
                afconvert_executable,
                "-f",
                "WAVE",
                "-d",
                f"LEI16@{sample_rate}",
                str(temp_aiff_path),
                str(output_file),
            ],
            check=True,
        )
    finally:
        temp_aiff_path.unlink(missing_ok=True)

    _validate_wav_file(output_file)
    return output_file


def _validate_wav_file(output_file: Path) -> None:
    try:
        with wave.open(str(output_file), "rb") as wav_file:
            if wav_file.getnframes() <= 0:
                raise RuntimeError(f"Generated WAV file has no audio frames: {output_file}")
    except wave.Error as exc:
        raise RuntimeError(f"Generated file is not a valid WAV file: {output_file}") from exc


def generate_speech(
    text: str,
    output_path: Union[str, Path],
    voice: Optional[str] = None,
    lang_code: str = DEFAULT_LANG_CODE,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    engine: str = "auto",
) -> Path:
    if not text.strip():
        raise ValueError("Text for TTS must not be empty.")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    normalized_engine = engine.lower()
    kokoro_error: Optional[Exception] = None

    if normalized_engine not in {"auto", "kokoro", "macos_say"}:
        raise ValueError("Unsupported TTS engine. Use one of: auto, kokoro, macos_say.")

    if normalized_engine in {"auto", "kokoro"}:
        try:
            return _generate_speech_with_kokoro(
                text=text,
                output_file=output_file,
                voice=voice,
                lang_code=lang_code,
                sample_rate=sample_rate,
            )
        except Exception as exc:
            kokoro_error = exc
            if normalized_engine == "kokoro":
                raise RuntimeError("Kokoro TTS failed.") from exc

    try:
        return _generate_speech_with_macos_say(
            text=text,
            output_file=output_file,
            voice=voice,
            sample_rate=sample_rate,
        )
    except Exception as exc:
        if kokoro_error is not None:
            raise RuntimeError(
                "Speech generation failed with both Kokoro and macOS 'say'."
            ) from kokoro_error
        raise RuntimeError("Speech generation failed with macOS 'say'.") from exc
