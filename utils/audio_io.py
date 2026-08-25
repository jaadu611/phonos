"""Audio I/O utilities"""

import subprocess
from pathlib import Path
from typing import Optional


def ensure_audio_format(input_path: str, target_format: str = "wav") -> Optional[str]:
    """Convert audio to a standard format if needed"""
    input_path = Path(input_path)

    if input_path.suffix.lower() == f".{target_format}":
        return str(input_path)

    output_path = input_path.with_suffix(f".{target_format}")

    try:
        subprocess.run(
            ["ffmpeg", "-i", str(input_path), "-y", str(output_path)],
            check=True,
            capture_output=True,
        )
        return str(output_path)
    except subprocess.CalledProcessError:
        return None


def convert_audio(input_path: str, output_path: str, sample_rate: int = 44100) -> bool:
    """Convert audio file using ffmpeg"""
    try:
        subprocess.run(
            [
                "ffmpeg", "-i", input_path,
                "-ar", str(sample_rate),
                "-y", output_path,
            ],
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False
