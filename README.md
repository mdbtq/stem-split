# stem-split

Splits audio files into stems — vocals, drums, bass, guitar, piano and other —
with neural source-separation models: a Mel-Band RoFormer for the vocals and
[Demucs](https://pypi.org/project/demucs/) for the rest, each stem taken from
the model that does it best. Each track becomes one folder of stems, with the
input's sample rate, channel count and length, ready for
[bpm-lock](https://github.com/mdbtq/bpm-lock)'s stem mode.

## Requirements

- Python 3.10–3.14 (3.12 is the tested version; PyTorch has no wheels for
  3.15 yet). Demucs, audio-separator, PyTorch and their dependencies are
  installed automatically. `audio-separator[cpu]` is the dependency; on an
  NVIDIA machine install `audio-separator[gpu]` alongside it for a faster RoFormer.
- `ffmpeg` / `ffprobe` on PATH (decoding/encoding any format)
- Apple Silicon (MPS) or an NVIDIA GPU (CUDA) is used automatically; the CPU
  works too, only slower.
- Network access on first use of a model. Demucs weights come from the
  Hugging Face hub and are cached in `~/.cache/huggingface`; the RoFormer
  weights (about 900 MB) come through audio-separator and are cached in
  `~/.cache/stem-split/audio-separator`.

## Install

```bash
uv tool install --python 3.12 git+https://github.com/mdbtq/stem-split@v0.3.0
```

Or with pipx: `pipx install --python python3.12 git+https://github.com/mdbtq/stem-split@v0.3.0`.
From a local checkout: `make build/install`.

## Usage

```bash
stem-split track.flac                       # -> track_stems/{vocals,drums,bass,guitar,piano,other}.flac
stem-split ~/Music/in -o ~/Music/stems      # whole folder (recursive)
stem-split track.mp3 -f wav                 # write the stems as WAV
stem-split track.wav -m htdemucs_ft         # Demucs only: 4 stems, about 2.5x faster
stem-split track.wav --exact-sum            # stems add up to the mix exactly
stem-split in/ --json report.json
```

Output per file:

```
track.flac: 6 stems [combined, mps] in 174.1 s (4:54.8 audio), residual -16.4 dB -> /path/track_stems
```

- `in … s` – time spent separating (model loading and file I/O not included).
- `residual` – level of what the stems miss: the mix minus the sum of all
  stems, relative to the mix. The models do not force their stems to add up to
  the input, so this is never zero. See [Notes](#notes). With `--exact-sum` the
  line adds `(added to other)`: the value is still what the model missed.
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

Add `--exact-sum` when the locked stems will be mixed back together and should
sound like the locked original.

## Exact sum (`--exact-sum`)

The residual — the mix minus the sum of the model's stems — is added to
`other`, so all stems together reproduce the input exactly. Nothing is lost
when the stems are summed again, at the price of a slightly busier `other`
stem: it now also carries whatever the models left out of the other
stems.

- The sum is exact up to the output format: within 24-bit rounding for WAV,
  AIFF and FLAC. Lossy formats (MP3, M4A, OGG) add their own error.
- When a stem would clip, the shared gain still applies; the stems then add up
  to the mix lowered by that gain.
- Needs a model with an `other` stem (every model listed below has one).

## Models

| Model | Stems | Notes |
|---|---|---|
| `combined` (default) | vocals, drums, bass, guitar, piano, other | See below. Slowest, best per stem |
| `htdemucs_ft` | drums, bass, other, vocals | A bag of four fine-tuned models, one per stem: clean drums, bass and vocals |
| `htdemucs` | drums, bass, other, vocals | One model. Faster, and its stems add up more closely to the mix |
| `htdemucs_6s` | + guitar, piano | One model. Its vocals, drums and bass are less clean than `htdemucs_ft`'s; piano is the weakest stem |

Any other name Demucs knows (e.g. `mdx_extra`) can be passed with `-m`.

### `combined`

1. **Vocals** — Mel-Band RoFormer (`vocals_mel_band_roformer.ckpt` by Kimberley
   Jensen, MIT), run through [audio-separator](https://pypi.org/project/audio-separator/).
2. The vocals are subtracted from the mix; both Demucs models run on what is left.
3. **Drums, bass** — `htdemucs_ft`.
4. **Guitar, piano, other** — `htdemucs_6s`.

Why this split, from listening to one 30 s excerpt (an ear test, not a
benchmark): the RoFormer vocals sounded best, with less kick and bass in them
(40–120 Hz bleed −54 dB against −45 dB for `htdemucs_ft`); `htdemucs_6s`'s
guitar sounded better than `htdemucs_ft`'s `other` used as a guitar stem, but
its vocals were worse.

Separated guitar tends to sound thin: measured on that excerpt, `htdemucs_6s`'s
guitar holds the 1–8 kHz range but its 300–1000 Hz body is 23 dB below the mix,
most of it ending up in `other`. Removing the vocals first does not change that.

## Options

| Option | Default | Meaning |
|---|---|---|
| `-m, --model` | `combined` | `combined` or a Demucs model |
| `--device` | `auto` | Demucs: `mps`, `cuda` or `cpu`; `auto` picks the first available in that order. RoFormer picks its own in the same order |
| `--shifts` | `1` | Demucs only: passes with a random time shift, averaged; slightly better, proportionally slower |
| `--exact-sum` | off | Add the residual to `other`, so the stems add up to the mix exactly |
| `-o, --out-dir` | next to input | Where the stem folders go |
| `--suffix` | `_stems` | Stem folder suffix; files inside such folders are skipped as input |
| `-f, --format` | same as input | `wav`, `flac`, `aiff`, `mp3`, `m4a`, `ogg` |
| `--json FILE` | – | Machine-readable report |

## Architecture

```
cli ─► pipeline.StemSplitter ─► audio_io (ffmpeg)
                             └─► separators.Separator   (protocol)
                                   ├─ DemucsSeparator    (one Demucs model)
                                   ├─ RoformerVocals     (audio-separator, vocals only)
                                   └─ CombinedSeparator  (RoFormer + two Demucs models)
```

Separators are injectable: the tests use fakes, and another backend can be
added without touching the pipeline. Both model families run at 44.1 kHz
stereo; the separators convert to that and back, so every stem has the input's
rate, channels and length. audio-separator works on files, so `RoformerVocals`
passes the audio through a private temp folder.

## Notes

- Expect clean drums, bass and vocals on most material. `other` is everything
  that is left (synths, pads, and guitar with the Demucs-only 4-stem models)
  and carries the most bleed. Dense
  electronic music with distorted basses separates less cleanly.
- The stems do not add up exactly to the mix. Measured on one 4:55 track
  at 48 kHz: −20.6 dB residual with `htdemucs_ft`; on a 30 s excerpt at the
  model's native 44.1 kHz, −23.2 dB with `htdemucs_ft` and −30.3 dB with
  `htdemucs`. The residual comes from the model, not from resampling. With
  `combined` it is larger (−16.4 dB on the same track), because its stems come
  from different models. Use `--exact-sum` when the stems have to sum back to
  the original.
- Speed and memory measured on an Apple M4 Pro (MPS) for the 4:55 track:
  `combined` took 174 s and about 5.7 GB of RAM, `htdemucs_ft` 67 s and about
  3.8 GB. Your machine will differ.
- Demucs, audio-separator and the default RoFormer weights are MIT-licensed.
  Check the licence of any other model weights before using the stems
  commercially.

## Tests

```bash
make test/all      # pipeline and CLI with a fake separator, no model needed
make test/model    # also runs htdemucs and combined on a short signal (downloads the weights once)
```

## License

MIT — see [LICENSE](LICENSE).
