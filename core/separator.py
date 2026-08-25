"""
Source separation using AudioSep for text-prompted extraction.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
import torch
import librosa
import soundfile as sf

AUDIOSEP_DIR = Path(__file__).resolve().parent.parent / "AudioSep"


class SourceSeparator:
    """Text-prompted audio source separation using AudioSep"""

    def __init__(self, device: str = "auto", checkpoint_path: Optional[str] = None):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self._loaded = False
        
        # Default checkpoint path
        if checkpoint_path is None:
            possible_paths = [
                Path("AudioSep/checkpoint/audiosep_base_4M_steps.ckpt"),
                Path("checkpoints/audiosep_base.pth"),
                Path("checkpoints/audiosep_base_4M_steps.ckpt"),
                AUDIOSEP_DIR / "checkpoint" / "audiosep_base_4M_steps.ckpt",
                Path.home() / ".cache" / "audioforge" / "audiosep_base.pth",
            ]
            for p in possible_paths:
                if p.exists():
                    checkpoint_path = str(p)
                    break
        
        self.checkpoint_path = checkpoint_path

    def load(self):
        """Load AudioSep model"""
        if self._loaded:
            return

        if self.checkpoint_path is None or not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(
                "AudioSep checkpoint not found. Download it:\n"
                "  wget https://huggingface.co/spaces/Audio-AGI/AudioSep/resolve/main/checkpoint/audiosep_base_4M_steps.ckpt -P AudioSep/checkpoint/\n"
            )

        try:
            if torch.cuda.is_available() and self.device == "cuda":
                torch.cuda.empty_cache()

            audio_sep_dir_str = str(AUDIOSEP_DIR.resolve())
            if audio_sep_dir_str not in sys.path:
                sys.path.insert(0, audio_sep_dir_str)

            # Isolate app modules from AudioSep internal utils/models
            app_modules = {}
            for k in list(sys.modules.keys()):
                if k == "utils" or k.startswith("utils.") or k == "models" or k.startswith("models."):
                    mod = sys.modules[k]
                    mod_file = getattr(mod, "__file__", "") or ""
                    if audio_sep_dir_str not in mod_file:
                        app_modules[k] = sys.modules.pop(k)

            from pipeline import build_audiosep
            
            config_yaml = AUDIOSEP_DIR / "config" / "audiosep_base.yaml"
            if not config_yaml.exists():
                config_yaml = Path("AudioSep/config/audiosep_base.yaml")

            self.model = build_audiosep(
                config_yaml=str(config_yaml),
                checkpoint_path=self.checkpoint_path,
                device=torch.device(self.device)
            )
            
            self._loaded = True

            # Restore app modules
            for k, v in app_modules.items():
                if k not in sys.modules:
                    sys.modules[k] = v
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load AudioSep: {e}\n"
                "Make sure AudioSep dependencies and checkpoints are installed."
            )

    def _separate_chunk_overlap_add(
        self,
        mixture: np.ndarray,
        conditions: torch.Tensor,
        chunk_size: float = 5.0,
        overlap: float = 0.5,
        sr: int = 32000
    ) -> np.ndarray:
        """
        Sliding window with Hann-weighted overlap-add crossfading.
        Prevents phase discontinuities and boundary pitch artifacts between chunks.
        """
        chunk_samples = int(chunk_size * sr)
        total_samples = len(mixture)

        if total_samples <= chunk_samples:
            input_dict = {
                "mixture": torch.tensor(mixture, dtype=torch.float32)[None, None, :].to(self.device),
                "condition": conditions,
            }
            sep = self.model.ss_model(input_dict)["waveform"]
            return sep.squeeze(0).squeeze(0).data.cpu().numpy()

        overlap_samples = int(overlap * chunk_samples)
        hop_size = chunk_samples - overlap_samples

        window = np.hanning(chunk_samples)

        output = np.zeros(total_samples, dtype=np.float32)
        weight_sum = np.zeros(total_samples, dtype=np.float32)

        for start in range(0, total_samples, hop_size):
            end = min(start + chunk_samples, total_samples)
            chunk = mixture[start:end]
            actual_len = len(chunk)

            if actual_len < chunk_samples:
                padded_chunk = np.pad(chunk, (0, chunk_samples - actual_len))
            else:
                padded_chunk = chunk

            input_dict = {
                "mixture": torch.tensor(padded_chunk, dtype=torch.float32)[None, None, :].to(self.device),
                "condition": conditions,
            }
            sep = self.model.ss_model(input_dict)["waveform"]
            sep_chunk = sep.squeeze(0).squeeze(0).data.cpu().numpy()[:actual_len]

            win = window[:actual_len]
            output[start:end] += sep_chunk * win
            weight_sum[start:end] += win

        mask = weight_sum > 1e-8
        output[mask] /= weight_sum[mask]
        return output

    def extract(self, audio_path: str, query: str) -> np.ndarray:
        """
        Extract a specific sound from audio using text query.
        
        Args:
            audio_path: Path to audio file
            query: Natural language description of sound to extract
                  e.g., "human speech", "helicopter", "gunfire", "explosion"
        
        Returns:
            numpy array of separated audio (the extracted sound only)
        """
        if not self._loaded:
            self.load()

        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.empty_cache()

        mixture, sr = self.load_audio(audio_path)

        with torch.no_grad():
            conditions = self.model.query_encoder.get_query_embed(
                modality='text',
                text=[query],
                device=torch.device(self.device)
            )

            sep_segment = self._separate_chunk_overlap_add(
                mixture=mixture,
                conditions=conditions,
                chunk_size=5.0,
                overlap=0.5,
                sr=sr
            )

        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.empty_cache()

        return sep_segment

    def extract_multiple(self, audio_path: str, queries: List[str]) -> Dict[str, np.ndarray]:
        """Extract multiple sounds from the same audio."""
        results = {}
        for query in queries:
            results[query] = self.extract(audio_path, query)
        return results

    def separate_speech_from_noise(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Specifically extract speech and return both speech and remaining noise."""
        speech = self.extract(audio_path, "human speech")
        original, sr = self.load_audio(audio_path)
        min_len = min(len(original), len(speech))
        noise = original[:min_len] - speech[:min_len]
        return speech[:min_len], noise

    def load_audio(self, path: str) -> Tuple[np.ndarray, int]:
        """Load audio file, return (waveform, sample_rate)"""
        waveform, sr = librosa.load(path, sr=32000, mono=True)
        return waveform, sr

    def save(self, audio: np.ndarray, output_path: str, sample_rate: int = 32000):
        """Save audio to file"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val * 0.95
        sf.write(output_path, audio, sample_rate)
