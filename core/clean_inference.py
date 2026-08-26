"""
AudioForge Pure Contiguous Audio Extraction Engine.
Guarantees ZERO phase comb-filtering, ZERO overlap-add ghost voices, and exact contiguous chunk reconstruction.
"""

from pathlib import Path
from typing import Union
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn


def extract_clean_speech(
    model: nn.Module,
    audio_input: Union[str, Path, np.ndarray],
    output_path: Union[str, Path],
    sr: int = 16000,
    segment_duration: float = 2.0,
    n_fft: int = 512,
    hop_length: int = 160,
    device: str = "cuda",
) -> np.ndarray:
    """
    Extracts speech using pure contiguous chunk processing (Zero Overlap-Add Flanger Distortion).
    """
    if isinstance(audio_input, (str, Path)):
        audio, _ = librosa.load(str(audio_input), sr=sr, mono=True)
    else:
        audio = audio_input.copy()

    chunk_samples = int(sr * segment_duration)
    enhanced_chunks = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i : i + chunk_samples]
            orig_len = len(chunk)
            if orig_len < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - orig_len))

            stft = librosa.stft(chunk.astype(np.float32), n_fft=n_fft, hop_length=hop_length)
            features = np.stack([stft.real, stft.imag])
            inp = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                pred = model(inp)

            pred_stft = pred[0, 0].float().cpu().numpy() + 1j * pred[0, 1].float().cpu().numpy()
            enhanced = librosa.istft(pred_stft, hop_length=hop_length)[:orig_len]
            enhanced_chunks.append(enhanced)

    output_audio = np.concatenate(enhanced_chunks)[: len(audio)]
    peak = np.max(np.abs(output_audio))
    if peak > 0:
        output_audio = (output_audio / peak) * 0.90

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), output_audio, sr)

    return output_audio
