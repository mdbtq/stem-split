"""Source separation: a Separator protocol plus the Demucs implementation."""
from __future__ import annotations

import sys
from typing import Protocol

import numpy as np

from .audio_io import Audio

DEFAULT_MODEL = "htdemucs_ft"
DEVICES = ("auto", "mps", "cuda", "cpu")


class SeparationError(RuntimeError):
    pass


class Separator(Protocol):
    name: str
    device: str

    def separate(self, audio: Audio) -> dict[str, Audio]:
        """Split `audio` into named stems with the input's sample rate, channels and length."""


def pick_device(requested: str = "auto") -> str:
    import torch
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class DemucsSeparator:
    """Demucs (Hybrid Transformer) models; weights are downloaded and cached on first use."""

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "auto", shifts: int = 1):
        try:
            from demucs.api import Separator as _Separator
        except ImportError as e:
            raise SeparationError(f"demucs is not installed: {e}") from e
        self.name = model
        self.device = pick_device(device)
        try:
            self._sep = _Separator(model=model, device=self.device, shifts=shifts,
                                   progress=sys.stderr.isatty())
        except Exception as e:  # unknown name, no network on first use, ...
            raise SeparationError(f"cannot load model '{model}': {e}") from e

    def separate(self, audio: Audio) -> dict[str, Audio]:
        import torch
        from demucs.audio import convert_audio

        rate, channels = self._sep.samplerate, self._sep.audio_channels
        wav = convert_audio(torch.from_numpy(audio.samples), audio.sample_rate, rate, channels)
        _, stems = self._sep.separate_tensor(wav)
        out = {}
        for name, x in stems.items():  # back to the input's rate, channels and length
            x = convert_audio(x.cpu(), rate, audio.sample_rate, audio.channels).numpy()
            x = np.pad(x, ((0, 0), (0, max(0, audio.frames - x.shape[1]))))[:, :audio.frames]
            out[name] = Audio(np.ascontiguousarray(x, dtype=np.float32), audio.sample_rate)
        return out
