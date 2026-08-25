"""
Stem amplification and manipulation.
Handles boosting, normalizing, and preparing stems for analysis.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path


class StemAmplifier:
    """Amplify and manipulate audio stems"""

    def __init__(self):
        self.peak_limit = 0.95  # Prevent clipping

    def boost(self, audio: np.ndarray, gain_db: float) -> np.ndarray:
        """
        Boost audio by specified gain in dB.

        Args:
            audio: Input audio array
            gain_db: Gain in decibels (positive = louder)

        Returns:
            Amplified audio array
        """
        # Convert dB to linear multiplier
        linear_gain = 10 ** (gain_db / 20)

        amplified = audio * linear_gain

        # Soft limiting to prevent harsh clipping
        amplified = self.soft_limit(amplified)

        return amplified

    def normalize(self, audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
        """Normalize audio to target peak level"""
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio * (target_peak / peak)
        return audio

    def soft_limit(self, audio: np.ndarray) -> np.ndarray:
        """Apply soft limiting (tanh) to prevent harsh digital clipping"""
        # tanh provides smooth saturation instead of hard clipping
        return np.tanh(audio * 1.5) / 1.5

    def spectral_enhance(self, audio: np.ndarray, sr: int, 
                         low_boost: float = 0.0, 
                         high_boost: float = 0.0) -> np.ndarray:
        """Apply spectral enhancement to bring out details"""
        stft = librosa.stft(audio)
        mag = np.abs(stft)
        phase = np.angle(stft)

        freqs = librosa.stft_frequencies(sr=sr, n_fft=2048)

        # Boost low frequencies
        if low_boost > 0:
            low_mask = freqs < 200
            mag[low_mask] *= 1 + low_boost

        # Boost high frequencies
        if high_boost > 0:
            high_mask = freqs > 4000
            mag[high_mask] *= 1 + high_boost

        # Reconstruct
        enhanced = mag * np.exp(1j * phase)
        return librosa.istft(enhanced)

    def compress(self, audio: np.ndarray, threshold: float = 0.5, 
                 ratio: float = 4) -> np.ndarray:
        """Dynamic range compression for better audibility"""
        # Simple compressor
        above_threshold = np.abs(audio) > threshold
        compressed = audio.copy()
        compressed[above_threshold] = np.sign(audio[above_threshold]) * (
            threshold + (np.abs(audio[above_threshold]) - threshold) / ratio
        )
        return compressed

    def prepare_for_analysis(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Full analysis preparation pipeline"""
        # 1. Normalize
        audio = self.normalize(audio, target_peak=0.8)

        # 2. Gentle compression for consistent listening level
        audio = self.compress(audio, threshold=0.3, ratio=2)

        # 3. Slight high-frequency enhancement for clarity
        audio = self.spectral_enhance(audio, sr, high_boost=0.3)

        return audio
