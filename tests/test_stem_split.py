import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from stem_split import audio_io, cli
from stem_split.audio_io import Audio
from stem_split.pipeline import StemSplitter

SR = 48000


def tone_with_clicks(duration: float = 4.0, sr: int = SR, channels: int = 2,
                     level: float = 0.5) -> Audio:
    """A 55 Hz tone plus a click every half second."""
    t = np.arange(int(duration * sr)) / sr
    y = 0.6 * np.sin(2 * np.pi * 55 * t)
    y[::sr // 2] += 1.0
    y = level * y / np.abs(y).max()
    return Audio(np.tile(y, (channels, 1)).astype(np.float32), sr)


class FakeSeparator:
    """Low band -> bass, the rest -> other, each scaled by `gain`; adds up to the mix × gain."""
    name, device = "fake", "cpu"

    def __init__(self, gain: float = 1.0):
        self.gain = gain

    def separate(self, audio: Audio) -> dict[str, Audio]:
        k = 64
        low = np.stack([np.convolve(ch, np.ones(k) / k, mode="same") for ch in audio.samples])
        return {"bass": Audio((low * self.gain).astype(np.float32), audio.sample_rate),
                "other": Audio(((audio.samples - low) * self.gain).astype(np.float32),
                               audio.sample_rate)}


class Pipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.src = self.tmp / "track.wav"
        audio_io.write(self.src, tone_with_clicks())

    def test_writes_one_file_per_stem_with_input_rate_and_length(self):
        r = StemSplitter(FakeSeparator()).process(self.src, self.tmp / "out", ".flac")
        self.assertEqual([Path(p).name for p in r.outputs], ["bass.flac", "other.flac"])
        src = audio_io.read(self.src)
        for p in r.outputs:
            stem = audio_io.read(Path(p))
            self.assertEqual((stem.sample_rate, stem.channels, stem.frames),
                             (SR, 2, src.frames))
        self.assertEqual(r.gain_db, 0.0)
        self.assertLess(r.residual_db, -60)   # the fake stems add up to the mix

    def test_loud_stems_get_one_shared_gain_instead_of_clipping(self):
        r = StemSplitter(FakeSeparator(gain=3.0)).process(self.src, self.tmp / "out", ".wav")
        stems = [audio_io.read(Path(p)).samples for p in r.outputs]
        self.assertLessEqual(max(np.abs(s).max() for s in stems), 1.0)
        self.assertLess(r.gain_db, 0)
        # Same gain on every stem: they still add up to the (scaled) mix.
        mix = audio_io.read(self.src).samples
        scale = 3.0 * 10 ** (r.gain_db / 20)
        self.assertLess(np.abs(sum(stems) - mix * scale).max(), 1e-3)

    def test_exact_sum_adds_the_residual_to_other(self):
        # Stems that miss 20 % of the mix: residual -14 dB, reported as the model's.
        splitter = StemSplitter(FakeSeparator(gain=0.8), exact_sum=True)
        r = splitter.process(self.src, self.tmp / "out", ".wav")
        self.assertAlmostEqual(r.residual_db, 20 * np.log10(0.2), delta=0.1)
        self.assertTrue(r.exact_sum)
        stems = {Path(p).stem: audio_io.read(Path(p)).samples for p in r.outputs}
        mix = audio_io.read(self.src).samples
        self.assertLess(np.abs(sum(stems.values()) - mix).max(), 1e-5)   # 24-bit rounding
        bass = FakeSeparator(gain=0.8).separate(audio_io.read(self.src))["bass"].samples
        self.assertLess(np.abs(stems["bass"] - bass).max(), 1e-5)        # only 'other' changed

    def test_exact_sum_needs_an_other_stem(self):
        class NoOther(FakeSeparator):
            def separate(self, audio):
                return {"bass": super().separate(audio)["bass"]}

        with self.assertRaisesRegex(ValueError, "no 'other' stem"):
            StemSplitter(NoOther(), exact_sum=True).process(self.src, self.tmp / "out", ".wav")


class Cli(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "in" / "sub").mkdir(parents=True)
        audio_io.write(self.tmp / "in" / "a.wav", tone_with_clicks())
        audio_io.write(self.tmp / "in" / "sub" / "b.flac", tone_with_clicks())
        patcher = mock.patch.object(cli, "build_separator", return_value=FakeSeparator())
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_folder_gives_one_stem_folder_per_track_and_rerun_skips_them(self):
        report = self.tmp / "report.json"
        self.assertEqual(cli.main([str(self.tmp / "in"), "--json", str(report)]), 0)
        self.assertTrue((self.tmp / "in" / "a_stems" / "bass.wav").is_file())
        self.assertTrue((self.tmp / "in" / "sub" / "b_stems" / "other.flac").is_file())
        self.assertEqual(len(json.loads(report.read_text())), 2)
        # The stem folders are output, not input: a second run splits the same two tracks.
        self.assertEqual(cli.main([str(self.tmp / "in"), "--json", str(report)]), 0)
        self.assertEqual(sorted(r["name"] for r in json.loads(report.read_text())),
                         ["a.wav", "b.flac"])

    def test_out_dir_and_format(self):
        out = self.tmp / "out"
        self.assertEqual(cli.main([str(self.tmp / "in" / "a.wav"), "-o", str(out), "-f", "mp3"]), 0)
        self.assertEqual(sorted(p.name for p in (out / "a_stems").iterdir()),
                         ["bass.mp3", "other.mp3"])

    def test_broken_file_fails_without_stopping_the_batch(self):
        (self.tmp / "in" / "broken.wav").write_bytes(b"not audio")
        self.assertEqual(cli.main([str(self.tmp / "in")]), 1)
        self.assertTrue((self.tmp / "in" / "a_stems").is_dir())


@unittest.skipUnless(os.environ.get("STEM_SPLIT_MODEL_TESTS"),
                     "set STEM_SPLIT_MODEL_TESTS=1 (downloads the htdemucs weights once)")
class DemucsModel(unittest.TestCase):
    def test_mono_48k_input_gives_mono_48k_stems_of_equal_length(self):
        from stem_split.separators import DemucsSeparator
        audio = tone_with_clicks(duration=6.0, channels=1)
        stems = DemucsSeparator("htdemucs").separate(audio)
        self.assertEqual(sorted(stems), ["bass", "drums", "other", "vocals"])
        for stem in stems.values():
            self.assertEqual((stem.sample_rate, stem.channels, stem.frames),
                             (SR, 1, audio.frames))


if __name__ == "__main__":
    unittest.main()
