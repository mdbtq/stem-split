# stem-split

Splits audio files into stems — drums, bass, vocals and other — with
[Demucs](https://pypi.org/project/demucs/), a neural source-separation
model. Each track becomes one folder of stems, with the input's sample rate,
channel count and length, ready for
[bpm-lock](https://github.com/mdbtq/bpm-lock)'s stem mode.

## Requirements

- Python 3.10–3.14 (3.12 is the tested version; PyTorch has no wheels for
  3.15 yet). Demucs, PyTorch and numpy are installed automatically (about 750 MB on macOS arm64).
- `ffmpeg` / `ffprobe` on PATH (decoding/encoding any format)
- Apple Silicon (MPS) or an NVIDIA GPU (CUDA) is used automatically; the CPU
  works too, only slower.
- Network access on first use of a model: the weights are downloaded from the
  Hugging Face hub and cached in `~/.cache/huggingface`.

## Install

```bash
uv tool install --python 3.12 git+https://github.com/mdbtq/stem-split@v0.1.0
```

Or with pipx: `pipx install --python python3.12 git+https://github.com/mdbtq/stem-split@v0.1.0`.
From a local checkout: `make build/install`.

## Usage

```bash
stem-split track.flac                       # -> track_stems/{drums,bass,other,vocals}.flac
stem-split ~/Music/in -o ~/Music/stems      # whole folder (recursive)
stem-split track.mp3 -f wav                 # write the stems as WAV
stem-split track.wav -m htdemucs_6s         # adds guitar and piano stems
stem-split in/ --json report.json
```

Output per file:

```
track.flac: 4 stems [htdemucs_ft, mps] in 67.2 s (4:54.8 audio), residual -20.6 dB -> /path/track_stems
```

- `in … s` – time spent separating (model loading and file I/O not included).
- `residual` – level of what the stems miss: the mix minus the sum of all
  stems, relative to the mix. Demucs does not force its stems to add up to the
  input, so this is never zero. See [Notes](#notes).
- `all stems lowered … dB` – only shown when a stem would exceed full scale.
  The same gain is applied to every stem, so their balance is unchanged.

Folders ending in the suffix (`_stems`) are output and are skipped as input, so
running the same command twice splits the same tracks again instead of
splitting the stems.

## With bpm-lock

Split first, on the original, then lock the stem folder: one analysis, one time
map, and every stem stays sample-aligned.

```bash
stem-split track.flac -f wav
bpm-lock track_stems --reference drums
```

## Models

| Model | Stems | Notes |
|---|---|---|
| `htdemucs_ft` (default) | drums, bass, other, vocals | A bag of four fine-tuned models, one per stem: the cleanest separation, slower than `htdemucs` |
| `htdemucs` | drums, bass, other, vocals | One model. Faster, and its stems add up more closely to the mix |
| `htdemucs_6s` | + guitar, piano | Piano is the weakest stem |

Any other name Demucs knows (e.g. `mdx_extra`) can be passed with `-m`.

## Options

| Option | Default | Meaning |
|---|---|---|
| `-m, --model` | `htdemucs_ft` | Demucs model |
| `--device` | `auto` | `mps`, `cuda` or `cpu`; `auto` picks the first available in that order |
| `--shifts` | `1` | Passes with a random time shift, averaged: slightly better, proportionally slower |
| `-o, --out-dir` | next to input | Where the stem folders go |
| `--suffix` | `_stems` | Stem folder suffix; files inside such folders are skipped as input |
| `-f, --format` | same as input | `wav`, `flac`, `aiff`, `mp3`, `m4a`, `ogg` |
| `--json FILE` | – | Machine-readable report |

## Architecture

```
cli ─► pipeline.StemSplitter ─► audio_io (ffmpeg)
                             └─► separators.Separator   (protocol; Demucs by default)
```

The separator is injectable: the tests use a filter-based fake, and another
backend can be added without touching the pipeline. Demucs runs at 44.1 kHz
stereo; `DemucsSeparator` converts to that and back, so every stem has the
input's rate, channels and length.

## Notes

- Expect clean drums, bass and vocals on most material. `other` is everything
  that is left (synths, guitars, pads) and carries the most bleed. Dense
  electronic music with distorted basses separates less cleanly.
- The stems do not add up exactly to the mix. Measured on one 4:55 track
  at 48 kHz: −20.6 dB residual with `htdemucs_ft`; on a 30 s excerpt at the
  model's native 44.1 kHz, −23.2 dB with `htdemucs_ft` and −30.3 dB with
  `htdemucs`. The residual comes from the model, not from resampling. Use
  `htdemucs` when the stems have to sum back to the original as closely as
  possible.
- Speed and memory measured on an Apple M4 Pro (MPS) with `htdemucs_ft`: the
  4:55 track took 67 s and about 3.8 GB of RAM. Your machine will differ.
- Demucs is MIT-licensed. Check the licence of the model weights you use
  before using the stems commercially.

## Tests

```bash
make test/all      # pipeline and CLI with a fake separator, no model needed
make test/model    # also runs htdemucs on a short signal (downloads it once)
```

## License

MIT — see [LICENSE](LICENSE).
