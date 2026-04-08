import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests


VOICE_LIST_URL = "https://raw.githubusercontent.com/rhasspy/piper/master/VOICES.md"
VOICE_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


@dataclass
class VoiceRef:
    voice_id: str
    model_path: str
    config_path: str


class PiperVoiceManager:
    def __init__(self, root_dir: Optional[str] = None):
        base = Path(root_dir) if root_dir else Path(__file__).resolve().parent
        self.root = base
        self.voices_dir = self.root / "models" / "piper_voices"
        self.voices_dir.mkdir(parents=True, exist_ok=True)

    def voice_paths(self, voice_id: str) -> Tuple[Path, Path]:
        model = self.voices_dir / f"{voice_id}.onnx"
        config = self.voices_dir / f"{voice_id}.onnx.json"
        return model, config

    def has_voice(self, voice_id: str) -> bool:
        model, config = self.voice_paths(voice_id)
        return model.exists() and config.exists()

    def ensure_voice(self, voice_id: str, timeout: int = 60) -> VoiceRef:
        model, config = self.voice_paths(voice_id)
        if not model.exists():
            self._download_voice_asset(voice_id, ".onnx", model, timeout=timeout)
        if not config.exists():
            self._download_voice_asset(voice_id, ".onnx.json", config, timeout=timeout)
        return VoiceRef(voice_id=voice_id, model_path=str(model), config_path=str(config))

    def _download_voice_asset(self, voice_id: str, suffix: str, target: Path, timeout: int = 60):
        rel = f"{voice_id}{suffix}"
        url = f"{VOICE_BASE_URL}/{rel}"
        with requests.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            with open(target, "wb") as out:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        out.write(chunk)

    def create_custom_voice(self, name: str, model_path: str, config_path: str, metadata: Dict) -> VoiceRef:
        safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", name.strip())
        target_dir = self.voices_dir / "custom"
        target_dir.mkdir(parents=True, exist_ok=True)

        dst_model = target_dir / f"{safe_name}.onnx"
        dst_config = target_dir / f"{safe_name}.onnx.json"
        dst_meta = target_dir / f"{safe_name}.meta.json"

        shutil.copy2(model_path, dst_model)
        shutil.copy2(config_path, dst_config)

        meta = {
            "name": safe_name,
            **metadata,
            "model_path": str(dst_model),
            "config_path": str(dst_config),
        }
        with open(dst_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return VoiceRef(voice_id=f"custom/{safe_name}", model_path=str(dst_model), config_path=str(dst_config))


def synthesize_to_wav(
    text: str,
    voice: VoiceRef,
    output_wav: str,
    speaker_id: int = 0,
    noise_scale: float = 0.667,
    length_scale: float = 1.0,
    noise_w: float = 0.8,
):
    output_path = Path(output_wav)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from piper import PiperVoice  # type: ignore

        piper_voice = PiperVoice.load(voice.model_path, config_path=voice.config_path)
        with open(output_path, "wb") as wav_file:
            piper_voice.synthesize(
                text,
                wav_file,
                speaker_id=speaker_id,
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_w=noise_w,
            )
        return str(output_path)
    except Exception:
        cmd = [
            "piper",
            "--model",
            voice.model_path,
            "--config",
            voice.config_path,
            "--output_file",
            str(output_path),
            "--speaker",
            str(speaker_id),
            "--noise_scale",
            str(noise_scale),
            "--length_scale",
            str(length_scale),
            "--noise_w",
            str(noise_w),
        ]
        proc = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Piper synthesis failed: {proc.stderr.decode('utf-8', errors='ignore')}")
        return str(output_path)
