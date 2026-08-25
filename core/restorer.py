"""
Audio restoration using VoiceFixer.
Reconstructs damaged speech, repairs harmonics, and removes residual noise.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import librosa
import soundfile as sf


class AudioRestorer:
    """Restores and reconstructs degraded audio using AI"""

    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self._loaded = False

    def load(self):
        """Load VoiceFixer model"""
        if self._loaded:
            return

        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.empty_cache()

        from voicefixer import VoiceFixer

        self.model = VoiceFixer()
        self._loaded = True

    def restore(self, audio: np.ndarray, sr: int = 44100, mode: int = 0) -> np.ndarray:
        """
        Restore degraded audio.
        Reconstructs missing frequency bands and damaged harmonics.

        Args:
            audio: Input audio array
            sr: Sample rate (VoiceFixer works natively at 44.1kHz)
            mode: 0 (Highest quality), 1 (Fast mode), 2 (Voice removal)

        Returns:
            Restored audio numpy array
        """
        if not self._loaded:
            self.load()

        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.empty_cache()

        # Normalize input to avoid clipping
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.9

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_input = os.path.join(tmp_dir, "input.wav")
            temp_output = os.path.join(tmp_dir, "output.wav")

            sf.write(temp_input, audio, sr)

            self.model.restore(
                input=temp_input,
                output=temp_output,
                cuda=(self.device == "cuda" and torch.cuda.is_available()),
                mode=mode,
            )

            restored, _ = librosa.load(temp_output, sr=sr, mono=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return restored

    def restore_file(self, input_path: str, output_path: str, mode: int = 0) -> str:
        """Restore audio file directly"""
        if not self._loaded:
            self.load()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        self.model.restore(
            input=input_path,
            output=output_path,
            cuda=(self.device == "cuda" and torch.cuda.is_available()),
            mode=mode,
        )

        return output_path
