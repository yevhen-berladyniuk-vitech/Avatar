from pathlib import Path

from pipeline import generate_speech

HARDCODED_TEXT = "Hello, this is a hardcoded TTS test for the avatar pipeline."
OUTPUT_WAV_PATH = Path("samples/output/tts_output.wav")


def main() -> None:
    generated_file = generate_speech(
        text=HARDCODED_TEXT,
        output_path=OUTPUT_WAV_PATH,
    )
    print(f"Generated WAV file: {generated_file.resolve()}")


if __name__ == "__main__":
    main()
