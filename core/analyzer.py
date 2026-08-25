"""
Audio analyzer using YAMNet for sound detection.
YAMNet is a pre-trained audio classifier from Google that 
recognizes 521 sound classes and runs efficiently on CPU.
"""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from typing import List, Dict


class AudioAnalyzer:
    """Detects what sounds are present in audio using YAMNet"""

    # YAMNet class names (subset - full list at https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv)
    # These are the most relevant for forensic/defence analysis
    FORENSIC_RELEVANT = {
        "Gunfire, gunfire": "gunfire",
        "Gunfire, generic": "gunfire",
        "Machine gun": "gunfire",
        "Artillery fire": "explosion",
        "Artillery, artillery fire": "explosion",
        "Explosion": "explosion",
        "Booming": "explosion",
        "Helicopter": "helicopter",
        "Rotorcraft, helicopter": "helicopter",
        "Aircraft": "aircraft",
        "Fixed-wing aircraft, airplane": "aircraft",
        "Jet engine": "jet_engine",
        "Engine": "engine",
        "Vehicle": "vehicle",
        "Truck": "truck",
        "Car vehicle, automobile motor vehicle": "car",
        "Speech": "speech",
        "Conversation": "speech",
        "Human voice": "speech",
        "Male speech, man speaking": "speech",
        "Female speech, woman speaking": "speech",
        "Whispering": "speech",
        "Chatter": "speech",
        "Crowd": "crowd",
        "Battle cry": "speech",
        "Shout": "speech",
        "Screaming": "speech",
        "Music": "music",
        "Siren": "siren",
        "Emergency vehicle": "siren",
        "Alarm": "alarm",
        "Wind": "wind",
        "Rain": "rain",
        "Thunderstorm": "thunder",
        "Rain on surface": "rain",
        "Noise": "noise",
        "Rustling leaves": "foliage",
        "Animal": "animal",
        "Dog": "dog",
        "Bird vocalization, bird call, bird song": "bird",
        "Insect": "insect",
        "Wild animals": "animal",
    }

    def __init__(self):
        self.model = None
        self.class_map = None
        self._loaded = False

    def load(self):
        """Load YAMNet model and class mappings"""
        if self._loaded:
            return

        # Suppress TF warnings
        tf.get_logger().setLevel("ERROR")
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        self.model = hub.load("https://tfhub.dev/google/yamnet/1")

        # Load class names
        import csv
        import io
        import urllib.request

        url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
        response = urllib.request.urlopen(url)
        class_map_bytes = response.read()
        self.class_map = list(csv.DictReader(io.StringIO(class_map_bytes.decode("utf-8"))))

        self._loaded = True

    def analyze(self, audio_path: str, threshold: float = 0.1) -> List[Dict]:
        """
        Analyze audio and return detected sounds with confidence scores.

        Args:
            audio_path: Path to audio file
            threshold: Minimum confidence to report (0.0 to 1.0)

        Returns:
            List of dicts: [{'sound': str, 'confidence': float, 'raw_label': str}]
        """
        if not self._loaded:
            self.load()

        # Load audio at YAMNet's expected sample rate
        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Run inference
        scores, embeddings, spectrogram = self.model(waveform)

        # Average scores across all frames
        mean_scores = np.mean(scores.numpy(), axis=0)

        # Map to readable names
        results = []
        for i, score in enumerate(mean_scores):
            if score > threshold:
                raw_label = self.class_map[i]["display_name"]
                friendly_name = self.FORENSIC_RELEVANT.get(raw_label, raw_label)

                # Skip generic categories
                if friendly_name in ("Speech", "Human voice"):
                    friendly_name = "speech"

                results.append(
                    {
                        "sound": friendly_name,
                        "confidence": float(score),
                        "raw_label": raw_label,
                    }
                )

        # Sort by confidence
        results.sort(key=lambda x: x["confidence"], reverse=True)

        # Deduplicate (multiple raw labels map to same friendly name)
        seen = set()
        deduped = []
        for r in results:
            if r["sound"] not in seen:
                seen.add(r["sound"])
                deduped.append(r)

        return deduped

    def analyze_timeline(self, audio_path: str, window_seconds: float = 1.0) -> List[Dict]:
        """
        Analyze audio over time, showing when each sound appears.

        Returns:
            List of dicts: [{'sound': str, 'start': float, 'end': float, 'confidence': float}]
        """
        if not self._loaded:
            self.load()

        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)

        scores, embeddings, spectrogram = self.model(waveform)
        scores = scores.numpy()

        # YAMNet processes in 0.96 second windows with 0.48s hop
        window_size = int(sr * 0.96)
        hop_size = int(sr * 0.48)

        timeline = []
        for frame_idx in range(scores.shape[0]):
            frame_scores = scores[frame_idx]
            time_start = (frame_idx * hop_size) / sr
            time_end = time_start + window_size / sr

            # Get top sound for this frame
            top_class = np.argmax(frame_scores)
            confidence = frame_scores[top_class]
            raw_label = self.class_map[top_class]["display_name"]
            friendly_name = self.FORENSIC_RELEVANT.get(raw_label, raw_label)

            if confidence > 0.2:
                timeline.append(
                    {
                        "sound": friendly_name,
                        "start": time_start,
                        "end": time_end,
                        "confidence": float(confidence),
                    }
                )

        # Merge consecutive same-sound detections
        merged = []
        for entry in timeline:
            if merged and merged[-1]["sound"] == entry["sound"]:
                if entry["start"] - merged[-1]["end"] < 2.0:  # Gap < 2 seconds
                    merged[-1]["end"] = entry["end"]
                    merged[-1]["confidence"] = max(
                        merged[-1]["confidence"], entry["confidence"]
                    )
            else:
                merged.append(entry.copy())

        return merged
