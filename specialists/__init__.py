"""
AudioForge Universal Speech Specialist & GAN Backbone.
"""

from .base_unet import HighResSpeechGenerator, MultiScaleSpeechDiscriminator, SpeechSpecialistUNet, SmallUNet
from .universal_speech.trainer import UniversalSpeechGANTrainer

__all__ = [
    "HighResSpeechGenerator",
    "MultiScaleSpeechDiscriminator",
    "SpeechSpecialistUNet",
    "SmallUNet",
    "UniversalSpeechGANTrainer",
]
