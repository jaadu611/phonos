"""
Source separation using AudioSep for text-prompted extraction.
AudioSep is a foundation model for open-domain audio source separation 
with natural language queries.
"""

import numpy as np
import torch
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional


class SourceSeparator:
    """Text-prompted audio source separation"""

    def __init__(self, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = None
        self._loaded = False

    def load(self):
        """Load AudioSep model"""
        if self._loaded:
            return

        try:
            # Try to import AudioSep
            from audiosep import AudioSep

            self.model = AudioSep(device=self.device)
            self._loaded = True

        except ImportError:
            raise ImportError(
                "AudioSep is not installed. Install it with:\n"
                "  git clone https://github.com/Audio-AGI/AudioSep\n"
                "  cd AudioSep\n"
                "  pip install -e ."
            )

    def extract(self, audio_path: str, query: str) -> np.ndarray:
        """
        Extract a specific sound from audio using text query.

        Args:
            audio_path: Path to audio file
            query: Natural language description of sound to extract

        Returns:
            numpy array of separated audio
        """
        if not self._loaded:
            self.load()

        # Load audio
        waveform, sr = self.load_audio(audio_path)

        # Run separation
        with torch.no_grad():
            separated = self.model.separate(audio=waveform, query=query)

        return separated

    def load_audio(self, path: str) -> tuple:
        """Load audio file, return (waveform, sample_rate)"""
        waveform, sr = librosa.load(path, sr=None, mono=True)
        return waveform, sr

    def save(self, audio: np.ndarray, output_path: str, sample_rate: int = 44100):
        """Save audio to file"""
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Normalize if needed
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val * 0.95

        sf.write(output_path, audio, sample_rate)

    def extract_all_stems(self, audio_path: str) -> dict:
        """Extract all detected stems from audio"""
        # First analyze what's in the audio
        from .analyzer import AudioAnalyzer

        analyzer = AudioAnalyzer()
        analyzer.load()
        detections = analyzer.analyze(audio_path)

        stems = {}
        for detection in detections:
            query = detection["sound"]
            stems[query] = self.extract(audio_path, query)

        return stems
