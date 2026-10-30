"""Command line interface: stem-split <files/dirs> [options]."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .pipeline import Report, StemSplitter
from .separators import DEFAULT_MODEL, DEVICES, DemucsSeparator, SeparationError, Separator

AUDIO_EXT = {".wav", ".aif", ".aiff", ".flac", ".mp3", ".m4a", ".ogg"}


def _collect(paths: list[Path], suffix: str) -> list[Path]:
    """Audio files given or found below the folders given; earlier output is skipped."""
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.is_file()
                            and f.suffix.lower() in AUDIO_EXT
                            and not any(d.name.endswith(suffix) for d in f.relative_to(p).parents))
        elif p.is_file():
            files.append(p)
        else:
            print(f"skip: {p} not found", file=sys.stderr)
    return files


def _output_dir(src: Path, out_dir: Path | None, suffix: str) -> Path:
    return (out_dir or src.parent) / f"{src.stem}{suffix}"


def _extension(src: Path, fmt: str | None) -> str:
    return f".{fmt.lstrip('.')}" if fmt else src.suffix


def build_separator(model: str, device: str, shifts: int) -> Separator:
    return DemucsSeparator(model, device, shifts)


def _clock(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:04.1f}"


def _format(r: Report) -> str:
    line = (f"{r.name}: {len(r.outputs)} stems [{r.model}, {r.device}] in {r.seconds:.1f} s "
            f"({_clock(r.duration_s)} audio), residual {r.residual_db:.1f} dB")
    if r.exact_sum:
        line += " (added to other)"
    if r.gain_db < 0:
        line += f", all stems lowered {-r.gain_db:.1f} dB to avoid clipping"
    return line + f" -> {Path(r.outputs[0]).parent}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stem-split",
        description="Split audio files into stems (drums, bass, vocals, other) with Demucs.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("inputs", nargs="+", type=Path, help="audio files and/or folders")
    p.add_argument("-m", "--model", default=DEFAULT_MODEL,
                   help=f"Demucs model: htdemucs_ft, htdemucs, htdemucs_6s … "
                        f"(default {DEFAULT_MODEL})")
    p.add_argument("--device", default="auto", choices=DEVICES,
                   help="compute device (default: mps, else cuda, else cpu)")
    p.add_argument("--shifts", type=int, default=1,
                   help="passes with a random time shift, averaged: slightly better, "
                        "proportionally slower (default 1)")
    p.add_argument("--exact-sum", action="store_true",
                   help="add what the model's stems miss to 'other', so all stems add up "
                        "to the mix exactly")
    p.add_argument("-o", "--out-dir", type=Path,
                   help="where the stem folders go (default: next to the input)")
    p.add_argument("--suffix", default="_stems",
                   help="stem folder suffix; files inside such folders are skipped as input")
    p.add_argument("-f", "--format", help="output format: wav, flac, aiff, mp3 … (default: same)")
    p.add_argument("--json", type=Path, help="write a JSON report")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    files = _collect(args.inputs, args.suffix)
    if not files:
        print("error: no audio files found", file=sys.stderr)
        return 2
    try:
        splitter = StemSplitter(build_separator(args.model, args.device, args.shifts),
                                exact_sum=args.exact_sum)
    except SeparationError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    reports: list[Report] = []
    failures = 0
    for src in files:   # one model on one device: files run one after another
        try:
            r = splitter.process(src, _output_dir(src, args.out_dir, args.suffix),
                                 _extension(src, args.format))
            reports.append(r)
            print(_format(r))
        except Exception as e:  # report and continue with the batch
            failures += 1
            print(f"{src.name}: FAILED - {e}", file=sys.stderr)

    if args.json:
        args.json.write_text(json.dumps([r.to_dict() for r in reports], indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
