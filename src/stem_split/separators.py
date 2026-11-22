"""Source separation: a Separator protocol, Demucs, RoFormer vocals, and a combination."""
from __future__ import annotations

import logging
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

import numpy as np

from . import audio_io
from .audio_io import Audio

COMBINED = "combined"
DEFAULT_MODEL = COMBINED
ROFORMER_MODEL = "vocals_mel_band_roformer.ckpt"   # Mel-Band RoFormer by Kimberley Jensen (MIT)
ROFORMER_CACHE = Path.home() / ".cache" / "stem-split" / "audio-separator"
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


def _conform(samples, rate: int, like: Audio) -> Audio:
    """Bring a model output (torch tensor) back to the rate, channels and length of `like`."""
    from demucs.audio import convert_audio

    x = convert_audio(samples.cpu(), rate, like.sample_rate, like.channels).numpy()
    x = np.pad(x, ((0, 0), (0, max(0, like.frames - x.shape[1]))))[:, :like.frames]
    return Audio(np.ascontiguousarray(x, dtype=np.float32), like.sample_rate)


def _hf_cached(model: str) -> bool:
    """Whether a Demucs model's definition is in the local Hugging Face cache already."""
    from demucs.hf import DEFAULT_NAMESPACE, hf_repo_name
    from huggingface_hub import try_to_load_from_cache

    namespace, _, name = model.rpartition("/")
    repo = f"{namespace or DEFAULT_NAMESPACE}/{hf_repo_name(name)}"
    return isinstance(try_to_load_from_cache(repo, f"{name}.yaml"), str)


@contextmanager
def _hf_offline(enabled: bool):
    """Hugging Face offline mode for the duration of the block. huggingface_hub reads
    the flag on every call, so this also works after the library has been imported."""
    from huggingface_hub import constants

    previous = constants.HF_HUB_OFFLINE
    constants.HF_HUB_OFFLINE = previous or enabled
    try:
        yield
    finally:
        constants.HF_HUB_OFFLINE = previous


class DemucsSeparator:
    """Demucs (Hybrid Transformer) models; weights are downloaded and cached on first use.

    A model already in the Hugging Face cache is loaded offline: without it,
    huggingface_hub checks the hub for a newer version on every run (and warns
    about unauthenticated requests).
    """

    def __init__(self, model: str = "htdemucs_ft", device: str = "auto", shifts: int = 1):
        try:
            from demucs.api import Separator as _Separator
        except ImportError as e:
            raise SeparationError(f"demucs is not installed: {e}") from e
        self.name = model
        self.device = pick_device(device)
        try:
            with _hf_offline(_hf_cached(model)):
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
        return {name: _conform(x, rate, audio) for name, x in stems.items()}


class RoformerVocals:
    """Vocals only, with a Mel-Band RoFormer model run through audio-separator.

    audio-separator works on files, so the audio goes through a private temp folder.
    It picks the device itself (MPS, CUDA or CPU).
    """

    def __init__(self, model: str = ROFORMER_MODEL):
        try:
            from audio_separator.separator import Separator as _Separator
        except ImportError as e:
            raise SeparationError(f"audio-separator is not installed: {e}") from e
        self.name = model
        self._tmp = tempfile.TemporaryDirectory(prefix="stem-split-")
        try:
            self._sep = _Separator(log_level=logging.WARNING, model_file_dir=str(ROFORMER_CACHE),
                                   output_dir=self._tmp.name, output_format="WAV",
                                   output_single_stem="Vocals",
                                   normalization_threshold=1.0)  # only scale a stem that clips
            self._sep.load_model(model)
        except Exception as e:
            raise SeparationError(f"cannot load model '{model}': {e}") from e
        self.device = str(self._sep.torch_device)

    def separate(self, audio: Audio) -> dict[str, Audio]:
        import torch

        src = Path(self._tmp.name) / "mix.wav"
        audio_io.write(src, audio)
        dst = Path(self._tmp.name) / "vocals.wav"
        self._sep.separate(str(src), custom_output_names={"Vocals": "vocals"})
        vocals = audio_io.read(dst)
        src.unlink()
        dst.unlink()
        return {"vocals": _conform(torch.from_numpy(vocals.samples), vocals.sample_rate, audio)}


class CombinedSeparator:
    """Each stem from the model best at it: vocals from RoFormer; drums and bass from
    htdemucs_ft and guitar, piano and other from htdemucs_6s, both run on the mix minus
    the vocals."""

    name = COMBINED

    def __init__(self, vocals: Separator, ft: Separator, six: Separator):
        self.vocals, self.ft, self.six = vocals, ft, six
        self.device = ft.device

    def separate(self, audio: Audio) -> dict[str, Audio]:
        vocals = self.vocals.separate(audio)["vocals"]
        rest = Audio(audio.samples - vocals.samples, audio.sample_rate)
        ft, six = self.ft.separate(rest), self.six.separate(rest)
        return {"vocals": vocals, "drums": ft["drums"], "bass": ft["bass"],
                "guitar": six["guitar"], "piano": six["piano"], "other": six["other"]}
