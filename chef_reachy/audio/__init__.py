"""Audio processing module for Chef Reachy."""

from chef_reachy.audio.config import AudioConfig
from chef_reachy.audio.tts import MeloTTSPlayer
from chef_reachy.audio.vad import SileroVAD, SpeechBuffer, SpeechState, VADConfig
from chef_reachy.audio.whisper import WhisperConfig, WhisperSTT

__all__ = [
    "AudioConfig",
    "MeloTTSPlayer",
    "SileroVAD",
    "SpeechBuffer",
    "SpeechState",
    "VADConfig",
    "WhisperConfig",
    "WhisperSTT",
]
