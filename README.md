# Avatar-poc

Single-command prototype for generating:

- a speech `.wav` from input text
- a SadTalker `.mp4` from one input portrait image plus that WAV

## Usage

If your base conda env is already active:

```bash
python3.11 run.py \
  --input samples/input/example-photo.png \
  --text "Hello from the Avatar prototype."
```

If you already have a WAV and just want to run the SadTalker leg:

```bash
python3.11 run.py \
  --input samples/input/example-photo.png \
  --audio path/to/input.wav
```

Outputs are written to `samples/output/` by default as:

- `<stem>.wav`
- `<stem>.mp4`

## SadTalker setup

The CLI expects SadTalker to be available in a separate conda env and checkout.

Supported defaults:

```bash
export SADTALKER_DIR=/Users/yevhen/PycharmProjects/SadTalker
export SADTALKER_CHECKPOINT_DIR=/Users/yevhen/PycharmProjects/SadTalker/checkpoints
```

It will run SadTalker through the `sadtalker` conda env by default. You can override that with:

```bash
python run.py \
  --input samples/input/example-photo.png \
  --text "Hello from the Avatar prototype." \
  --sadtalker-conda-env sadtalker
```

## Notes

- TTS defaults to `--tts-engine auto`, which tries Kokoro first and falls back to macOS `say`.
- You can force the fallback path with `--tts-engine macos_say`.
- You can skip TTS entirely with `--audio <existing.wav>`.
- You can keep SadTalker's temporary directory with `--keep-sadtalker-temp`.
