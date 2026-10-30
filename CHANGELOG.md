# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-10-30

### Added
- `--exact-sum`: adds the residual (mix minus the sum of the model's stems) to
  the `other` stem, so all stems add up to the input exactly.
- JSON report: `exact_sum`. `residual_db` stays the model's residual.

## [0.1.0] - 2026-10-30

### Added
- `stem-split` command: splits audio files and folders into stems with Demucs
  (`htdemucs_ft` by default; `-m` selects another model). One folder per track,
  `<track>_stems/<stem>.<ext>`, with the input's sample rate, channels and length.
- `--device` (MPS, CUDA or CPU, picked automatically), `--shifts`, `-o`,
  `--suffix`, `-f` and `--json`.
- Report per file: separation time, residual (mix minus the sum of the stems)
  and the shared gain applied when a stem would clip.
