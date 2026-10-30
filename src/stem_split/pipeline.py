"""One file in, one folder of stems out."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import audio_io
from .audio_io import Audio
from .separators import Separator


@dataclass
class Report:
    name: str
    input: str
    outputs: list[str]
    model: str
    device: str
    duration_s: float   # length of the input
    seconds: float      # time spent separating
    gain_db: float      # gain applied to every stem to stay below full scale (0 = none)
    residual_db: float  # level of mix minus the sum of the model's stems, relative to the mix
    exact_sum: bool     # residual added to the 'other' stem, so the stems add up to the mix

    def to_dict(self) -> dict:
        return asdict(self)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def _db(ratio: float) -> float:
    return 20 * np.log10(max(ratio, 1e-10))


class StemSplitter:
    def __init__(self, separator: Separator, exact_sum: bool = False):
        self.separator = separator
        self.exact_sum = exact_sum

    def process(self, src: Path, out_dir: Path, ext: str) -> Report:
        """Write one file per stem, `<out_dir>/<stem><ext>`."""
        audio = audio_io.read(src)
        start = time.perf_counter()
        stems = self.separator.separate(audio)
        seconds = time.perf_counter() - start

        residual = audio.samples - sum(s.samples for s in stems.values())
        if self.exact_sum:
            if "other" not in stems:
                raise ValueError(f"model '{self.separator.name}' has no 'other' stem")
            other = stems["other"]
            stems["other"] = Audio(other.samples + residual, other.sample_rate)
        # One gain for all stems, so they still add up to the mix (scaled) instead of clipping.
        peak = max(float(np.abs(s.samples).max()) for s in stems.values())
        gain = 1.0 / peak if peak > 1.0 else 1.0
        outputs = []
        for name, stem in stems.items():
            dst = out_dir / f"{name}{ext}"
            audio_io.write(dst, Audio(stem.samples * gain, stem.sample_rate))
            outputs.append(str(dst))
        return Report(
            name=src.name, input=str(src), outputs=outputs,
            model=self.separator.name, device=self.separator.device,
            duration_s=audio.duration, seconds=seconds, gain_db=_db(gain),
            residual_db=_db(_rms(residual) / max(_rms(audio.samples), 1e-12)),
            exact_sum=self.exact_sum,
        )
