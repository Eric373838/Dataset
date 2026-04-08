# ComfyUI Piper Nodepack

A ComfyUI custom node pack for **Piper TTS** with:

- automatic voice bootstrapping ("preinstalled" managed voices)
- text-to-speech generation
- custom voice profile saving
- fine-tuning workspace creation + trainer command execution hooks

> Note: Piper fine-tuning depends on external trainer tooling (`piper_train` / VITS training stack). This nodepack exposes the workflow in ComfyUI and can execute trainer commands, but you still need the trainer dependencies installed.

## Features

### 1) Managed voices (auto-usable)
- Automatically ensures requested voices exist locally.
- Downloads official `.onnx` + `.onnx.json` voice assets on demand.
- Keeps voices in `models/piper_voices/` inside this nodepack by default.

### 2) Generation (TTS)
- Generates WAV files from text.
- Uses the Piper Python API when available.
- Falls back to `piper` CLI when Python package API is unavailable.

### 3) Fine-tuning workflow
- Builds a dataset/fine-tune workspace from `{audio,text}` pairs.
- Writes a complete `manifest.jsonl` for trainer use.
- Runs a custom fine-tune command template.

### 4) Custom voice saving
- Save a trained `.onnx` + `.onnx.json` pair as a reusable custom voice profile.
- Stores metadata (`meta.json`) for future traceability.

## Install in ComfyUI

1. Copy this folder into:
   `ComfyUI/custom_nodes/comfyui-piper-nodepack`
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Restart ComfyUI.

## Dependencies

- `requests`
- `numpy`
- `scipy`
- `piper-tts` (optional but recommended)

If `piper-tts` is unavailable, generation can still work through the `piper` CLI.

## Available Nodes

- `Piper Ensure Voice`
- `Piper Generate Speech`
- `Piper Build Finetune Workspace`
- `Piper Run Finetune`
- `Piper Save Custom Voice`

## Example flow

1. `Piper Ensure Voice`
   - `voice_id`: `en_US-amy-medium`
2. `Piper Generate Speech`
   - `text`: your script
   - `voice_ref`: from previous node
3. (Optional) `Piper Build Finetune Workspace`
4. (Optional) `Piper Run Finetune`
5. `Piper Save Custom Voice`

## Fine-tune command template example

Use this in `Piper Run Finetune`:

```bash
python -m piper_train \
  --dataset-dir {workspace} \
  --output-dir {output_dir} \
  --checkpoint-epochs 5
```

Template variables:
- `{workspace}`
- `{manifest}`
- `{output_dir}`

## Voice catalog source

By default the manager uses:
`https://raw.githubusercontent.com/rhasspy/piper/master/VOICES.md`

And downloads voice files from:
`https://huggingface.co/rhasspy/piper-voices/resolve/main`

## License

MIT (for this nodepack code). Voice models have their own upstream licenses.
