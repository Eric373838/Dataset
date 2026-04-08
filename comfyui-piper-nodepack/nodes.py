import json
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from .piper_manager import PiperVoiceManager, synthesize_to_wav


NODE_CATEGORY = "audio/piper_tts"


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


class PiperEnsureVoiceNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voice_id": ("STRING", {"default": "en_US-amy-medium", "multiline": False}),
                "root_dir": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("voice_ref", "model_path", "config_path")
    FUNCTION = "run"
    CATEGORY = NODE_CATEGORY

    def run(self, voice_id: str, root_dir: str):
        manager = PiperVoiceManager(root_dir=root_dir or None)
        voice = manager.ensure_voice(voice_id.strip())
        voice_ref = json.dumps(
            {
                "voice_id": voice.voice_id,
                "model_path": voice.model_path,
                "config_path": voice.config_path,
                "resolved_at": _now_iso(),
            }
        )
        return (voice_ref, voice.model_path, voice.config_path)


class PiperGenerateSpeechNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello from Piper inside ComfyUI."}),
                "voice_ref": ("STRING", {"multiline": False}),
                "output_wav": ("STRING", {"default": "outputs/piper/generated.wav", "multiline": False}),
                "speaker_id": ("INT", {"default": 0, "min": 0, "max": 999}),
                "noise_scale": ("FLOAT", {"default": 0.667, "min": 0.0, "max": 5.0, "step": 0.001}),
                "length_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.001}),
                "noise_w": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 5.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("wav_path", "status")
    FUNCTION = "run"
    CATEGORY = NODE_CATEGORY

    def run(
        self,
        text: str,
        voice_ref: str,
        output_wav: str,
        speaker_id: int,
        noise_scale: float,
        length_scale: float,
        noise_w: float,
    ):
        data = json.loads(voice_ref)
        wav_path = synthesize_to_wav(
            text=text,
            voice=type("VoiceRef", (), data)(),
            output_wav=output_wav,
            speaker_id=speaker_id,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_w=noise_w,
        )
        status = f"generated:{wav_path}"
        return wav_path, status


class PiperBuildFinetuneWorkspaceNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workspace": ("STRING", {"default": "outputs/piper/finetune_workspace", "multiline": False}),
                "items_json": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": '[{"audio":"/path/to/a.wav","text":"example transcript"}]',
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("workspace", "manifest", "item_count")
    FUNCTION = "run"
    CATEGORY = NODE_CATEGORY

    def run(self, workspace: str, items_json: str):
        workspace_path = Path(workspace)
        audio_dir = workspace_path / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = workspace_path / "manifest.jsonl"
        items: List[Dict] = json.loads(items_json)

        count = 0
        with open(manifest_path, "w", encoding="utf-8") as man:
            for idx, item in enumerate(items):
                src_audio = Path(item["audio"]) 
                text = str(item["text"]).strip()
                dst = audio_dir / f"sample_{idx:06d}{src_audio.suffix or '.wav'}"
                dst.write_bytes(src_audio.read_bytes())

                entry = {
                    "audio_filepath": str(dst.resolve()),
                    "text": text,
                    "duration": item.get("duration", 0),
                }
                man.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

        return (str(workspace_path.resolve()), str(manifest_path.resolve()), count)


class PiperRunFinetuneNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workspace": ("STRING", {"default": "outputs/piper/finetune_workspace", "multiline": False}),
                "output_dir": ("STRING", {"default": "outputs/piper/finetune_runs/latest", "multiline": False}),
                "command_template": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "python -m piper_train --dataset-dir {workspace} --output-dir {output_dir}",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("command", "returncode", "log_path")
    FUNCTION = "run"
    CATEGORY = NODE_CATEGORY

    def run(self, workspace: str, output_dir: str, command_template: str):
        workspace_path = Path(workspace).resolve()
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        manifest = workspace_path / "manifest.jsonl"

        command = command_template.format(
            workspace=str(workspace_path),
            manifest=str(manifest),
            output_dir=str(output_path),
        )
        args = shlex.split(command)

        proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)

        log_path = output_path / "finetune.log"
        log_path.write_text(proc.stdout, encoding="utf-8")

        return command, proc.returncode, str(log_path)


class PiperSaveCustomVoiceNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "my_custom_voice", "multiline": False}),
                "model_path": ("STRING", {"multiline": False}),
                "config_path": ("STRING", {"multiline": False}),
                "root_dir": ("STRING", {"default": "", "multiline": False}),
                "notes": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("voice_ref", "saved_model", "saved_config")
    FUNCTION = "run"
    CATEGORY = NODE_CATEGORY

    def run(self, name: str, model_path: str, config_path: str, root_dir: str, notes: str):
        manager = PiperVoiceManager(root_dir=root_dir or None)
        voice = manager.create_custom_voice(
            name=name,
            model_path=model_path,
            config_path=config_path,
            metadata={
                "saved_at": _now_iso(),
                "notes": notes,
            },
        )
        voice_ref = json.dumps(
            {
                "voice_id": voice.voice_id,
                "model_path": voice.model_path,
                "config_path": voice.config_path,
                "resolved_at": _now_iso(),
            }
        )
        return voice_ref, voice.model_path, voice.config_path


NODE_CLASS_MAPPINGS = {
    "PiperEnsureVoice": PiperEnsureVoiceNode,
    "PiperGenerateSpeech": PiperGenerateSpeechNode,
    "PiperBuildFinetuneWorkspace": PiperBuildFinetuneWorkspaceNode,
    "PiperRunFinetune": PiperRunFinetuneNode,
    "PiperSaveCustomVoice": PiperSaveCustomVoiceNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PiperEnsureVoice": "Piper Ensure Voice",
    "PiperGenerateSpeech": "Piper Generate Speech",
    "PiperBuildFinetuneWorkspace": "Piper Build Finetune Workspace",
    "PiperRunFinetune": "Piper Run Finetune",
    "PiperSaveCustomVoice": "Piper Save Custom Voice",
}
