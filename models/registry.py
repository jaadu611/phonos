"""
Model registry: handles downloading, caching, and loading models.
Models are cached in ~/.cache/audioforge/
"""

import os
from pathlib import Path
from typing import Dict, Optional


class ModelRegistry:
    """Central registry for all models used by AudioForge"""

    CACHE_DIR = Path.home() / ".cache" / "audioforge"

    MODELS = {
        "yamnet": {
            "url": "https://tfhub.dev/google/yamnet/1",
            "type": "tensorflow-hub",
            "size_mb": 4,
            "description": "Audio classification (521 sound classes)",
        },
        "audiosep": {
            "repo": "https://github.com/Audio-AGI/AudioSep",
            "type": "custom",
            "size_mb": 500,
            "description": "Text-prompted source separation",
        },
    }

    def __init__(self):
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._loaded_models: Dict[str, object] = {}

    def get_model(self, name: str):
        """Get a model by name, loading if necessary"""
        if name in self._loaded_models:
            return self._loaded_models[name]

        if name not in self.MODELS:
            raise ValueError(f"Unknown model: {name}. Available: {list(self.MODELS.keys())}")

        model_config = self.MODELS[name]

        if model_config["type"] == "tensorflow-hub":
            import tensorflow_hub as hub
            model = hub.load(model_config["url"])
        elif model_config["type"] == "custom":
            # Custom loading logic per model
            if name == "audiosep":
                model = self._load_audiosep()

        self._loaded_models[name] = model
        return model

    def _load_audiosep(self):
        """Load AudioSep model"""
        try:
            from audiosep import AudioSep
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            return AudioSep(device=device)
        except ImportError:
            raise ImportError(
                "AudioSep not installed. See https://github.com/Audio-AGI/AudioSep"
            )

    def list_models(self):
        """List available models"""
        for name, config in self.MODELS.items():
            print(f"  {name}: {config['description']} ({config['size_mb']}MB)")
