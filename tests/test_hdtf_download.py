"""HDTF range downloads, real media trimming, recovery, and shared CLI routing."""

import contextlib
import io
import json
import re
import shutil
import subprocess
import threading
import zipfile
from http.client import IncompleteRead
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch
import unittest

import hdtf_download as archive_io
import train
import train_downvid as pipeline
from tests.test_train_downvid import PipelineFixture


class RangeArchiveTests(unittest.TestCase):
    def archive_bytes(self):
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("unused.mp4", b"x" * (3 * 1024 * 1024))
            archive.writestr("selected.mp4", b"selected video", compress_type=zipfile.ZIP_DEFLATED)
        return output.getvalue()

    def response(self, payload, start, end, status=206, etag='"archive-v1"'):
        result = io.BytesIO(payload[start:end + 1])
        result.status = status
        result.headers = {"Content-Range": f"bytes {start}-{end}/{len(payload)}", "ETag": etag}
        return result

    def test_ranges_read_only_selected_member_and_index(self):
        payload, requests = self.archive_bytes(), []

        def respond(request, **kwargs):
            start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range")).groups())
            requests.append((start, end))
            return self.response(payload, start, end)

        with patch.object(archive_io, "urlopen", side_effect=respond):
            with archive_io.open_archive("https://example.test/videos.zip", 30, 1) as archive:
                self.assertEqual(archive.read("selected.mp4"), b"selected video")
        self.assertLess(sum(end - start + 1 for start, end in requests), 4096)

    def test_ignored_ranges_and_changed_archives_stop(self):
        payload = self.archive_bytes()
        with patch.object(archive_io, "urlopen", return_value=self.response(payload, 0, 0, status=200)):
            with self.assertRaisesRegex(archive_io.ArchiveAccessError, "byte ranges"):
                archive_io.HTTPRangeReader("https://example.test/a.zip", 30, 0)
        with patch.object(archive_io, "urlopen", side_effect=[
            self.response(payload, 0, 0), self.response(payload, 0, 1, etag='"changed"'),
        ]):
            with archive_io.HTTPRangeReader("https://example.test/a.zip", 30, 0) as reader:
                with self.assertRaisesRegex(archive_io.ArchiveAccessError, "changed"):
                    reader.read(2)

    def test_independent_reader_rejects_change_from_listing_identity(self):
        payload = self.archive_bytes()
        with patch.object(archive_io, "urlopen", return_value=self.response(payload, 0, 0, etag='"changed"')):
            with self.assertRaisesRegex(archive_io.ArchiveAccessError, "changed"):
                archive_io.HTTPRangeReader("https://example.test/a.zip", 30, 0,
                                           identity=(len(payload), '"archive-v1"'))

    def test_transient_http_retry_and_nonretryable_access_failure(self):
        payload = self.archive_bytes()
        with patch.object(archive_io, "urlopen", side_effect=[
            HTTPError("https://example.test", 503, "unavailable", {}, None),
            self.response(payload, 0, 0),
        ]) as request, patch.object(archive_io.time, "sleep"):
            with archive_io.HTTPRangeReader("https://example.test/a.zip", 30, 1):
                pass
            self.assertEqual(request.call_count, 2)
        with patch.object(archive_io, "urlopen", side_effect=HTTPError(
            "https://example.test?secret=hidden", 403, "forbidden", {}, None,
        )) as request:
            with self.assertRaisesRegex(archive_io.ArchiveAccessError, "HTTP 403") as caught:
                archive_io.HTTPRangeReader("https://example.test/a.zip", 30, 4)
            self.assertEqual(request.call_count, 1)
            self.assertNotIn("secret", str(caught.exception))

    def test_interrupted_response_retries_same_byte_range(self):
        payload = self.archive_bytes()
        interrupted = self.response(payload, 0, 0)
        with patch.object(interrupted, "read", side_effect=IncompleteRead(b"", 1)), \
                patch.object(archive_io, "urlopen", side_effect=[
                    interrupted, self.response(payload, 0, 0),
                ]) as request, patch.object(archive_io.time, "sleep"):
            with archive_io.HTTPRangeReader("https://example.test/a.zip", 30, 1) as reader:
                self.assertEqual(reader.size, len(payload))
            self.assertEqual([call.args[0].get_header("Range") for call in request.call_args_list],
                             ["bytes=0-0", "bytes=0-0"])


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class HDTFTests(PipelineFixture):
    def setUp(self):
        super().setUp()
        self.source = self.root / "source.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
            "testsrc2=size=32x32:rate=25:duration=2", "-f", "lavfi", "-i",
            "sine=frequency=440:duration=2", "-c:v", "libx264", "-c:a", "aac", str(self.source),
        ], check=True, timeout=30)
        self.zip_path = self.root / "hdtf.zip"
        with zipfile.ZipFile(self.zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("README.txt", "ignored")
            archive.writestr("videos/00-broken.mp4", b"invalid video")
            archive.write(self.source, "videos/02-valid.mp4")
            archive.write(self.source, "videos/01-valid.mp4")
            archive.write(self.source, "videos/03-valid.mp4")

    def hdtf_args(self, *flags):
        return self.args("--dataset", "hdtf", "--hdtf-archive", str(self.zip_path), *flags)

    def test_caps_audio_and_frames_and_separates_caches(self):
        paths = []
        for limit, frames in ((None, 50), (12, 12), (1, 1), (100, 50)):
            with self.subTest(limit=limit):
                args = self.hdtf_args(*([] if limit is None else ["--download-max-frames", str(limit)]))
                with contextlib.closing(pipeline.iter_hdtf(args)) as rows:
                    next(rows)
                    row = next(rows)
                    video = pipeline.download_hdtf_clip(row["clip"], args, row["archive"], row["member"])
                    with patch.object(pipeline, "copy_video", side_effect=AssertionError("cache missed")):
                        self.assertEqual(pipeline.download_hdtf_clip(
                            row["clip"], args, row["archive"], row["member"]), video)
                result = subprocess.run([
                    "ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(video),
                ], check=True, capture_output=True, text=True, timeout=30)
                streams = {s["codec_type"]: s for s in json.loads(result.stdout)["streams"]}
                self.assertEqual(int(streams["video"]["nb_read_frames"]), frames)
                self.assertAlmostEqual(float(streams["audio"]["duration"]), frames / 25, delta=.06)
                paths.append(video)
        self.assertEqual(len(set(paths)), 4)

    def prepare_fixture(self, args):
        grouper = pipeline.load_grouper(pipeline.DEFAULT_GROUPER)
        words = [grouper.WordSpan("hi", 0, 2, 0.04, 0.3)]
        with patch.object(pipeline, "load_grouper", return_value=grouper), \
                patch.object(grouper, "WhisperModel"), \
                patch.object(grouper, "transcribe_words", return_value=("hi", words)) as asr, \
                patch.object(pipeline, "download_clip", side_effect=AssertionError("YouTube called")), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            pipeline.prepare(args)
        return asr.call_count

    def test_exact_n_replaces_bad_video_and_reuses_alignments(self):
        args = self.hdtf_args("-N", "2", "--download-max-frames", "12")
        self.assertEqual(self.prepare_fixture(args), 2)
        records = train.load_manifest(args.manifest)
        self.assertEqual(len(records), 2)
        self.assertEqual(len({row["video"] for row in records}), 2)
        report = pipeline.read_cache(self.root / "run_report.json")
        self.assertEqual(report["dataset"], "hdtf")
        self.assertEqual(report["attempts"], 3)
        self.assertEqual([r["id"] for r in report["results"]],
                         ["videos/01-valid.mp4", "videos/02-valid.mp4"])
        self.assertEqual(len(report["failures"]), 1)
        self.assertEqual(self.prepare_fixture(args), 0)

    def test_video_only_pretraining_replaces_bad_members_without_whisper(self):
        args = self.hdtf_args("-N", "2", "--download-max-frames", "3")
        args.manifest = self.root / "pretrain.jsonl"
        with patch.object(pipeline, "load_grouper", side_effect=AssertionError("Whisper was loaded")), \
                patch.object(pipeline, "download_clip", side_effect=AssertionError("YouTube called")), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            pipeline.prepare(args, video_only=True)
        rows = train.load_manifest(args.manifest, video_only=True)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(set(row) == {"video", "source_key"} for row in rows))
        report = pipeline.read_cache(self.root / "pretrain_report.json")
        self.assertEqual((report["usable"], report["attempts"]), (2, 3))

    def test_parallel_http_members_use_independent_readers_and_close_them(self):
        args = self.hdtf_args("-N", "2", "--start-index", "1", "--download-workers", "2",
                              "--hdtf-archive", "https://example.test/videos.zip")
        payload = self.zip_path.read_bytes()
        barrier = threading.Barrier(2)
        readers = []
        copy = pipeline.copy_video

        def respond(request, **kwargs):
            start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range")).groups())
            return RangeArchiveTests().response(payload, start, end)

        def transfer(archive, member, output, timeout):
            readers.append(archive.fp)
            barrier.wait(timeout=5)
            return copy(archive, member, output, timeout)

        with patch.object(archive_io, "urlopen", side_effect=respond), \
                patch.object(pipeline, "copy_video", side_effect=transfer), \
                contextlib.redirect_stdout(io.StringIO()):
            pipeline.prepare(args, video_only=True)
        self.assertEqual(len(readers), 2)
        self.assertIsNot(readers[0], readers[1])
        self.assertTrue(all(reader.closed for reader in readers))
        for row in train.load_manifest(args.manifest, video_only=True):
            self.assertEqual(Path(row["video"]).read_bytes(), self.source.read_bytes())

    def test_independent_archive_rejects_member_changed_after_listing(self):
        args = self.hdtf_args("--start-index", "1")
        with contextlib.closing(pipeline.iter_hdtf(args)) as rows:
            next(rows)
            selected = next(rows)
        with zipfile.ZipFile(self.zip_path, "w") as archive:
            archive.writestr(selected["clip"].member, b"replacement")
        with self.assertRaisesRegex(archive_io.ArchiveAccessError, "changed"):
            pipeline.download_hdtf_clip(selected["clip"], args, member=selected["member"])
        self.assertIsNone(pipeline.cached_download(selected["clip"], args))

    def test_shortfall_preserves_final_manifest_and_start_index(self):
        args = self.hdtf_args("-N", "2", "--start-index", "3", "--download-max-frames", "12")
        args.manifest.write_text("previous manifest\n")
        with self.assertRaisesRegex(RuntimeError, "Only 1/2"):
            self.prepare_fixture(args)
        self.assertEqual(args.manifest.read_text(), "previous manifest\n")
        report = pipeline.read_cache(self.root / "run_report.json")
        self.assertEqual(report["attempts"], 1)
        self.assertEqual(report["results"][0]["id"], "videos/03-valid.mp4")
        self.assertEqual(len(train.load_manifest(str(args.manifest) + ".partial")), 1)

    def test_hdtf_validation_skips_youtube_dependencies_and_saved_cookies(self):
        parser = pipeline.build_argparser()
        args = parser.parse_args(["--dataset", "hdtf", "-N", "1"])
        with patch.object(pipeline.importlib.util, "find_spec", side_effect=lambda name: object()
                          if name == "faster_whisper" else None) as dependencies, \
                patch.object(pipeline, "validate_download_runtimes") as runtime, \
                patch.object(pipeline, "use_saved_login") as cookies:
            pipeline.validate_args(parser, args)
        self.assertEqual(args.work_dir, pipeline.ROOT / "data" / "hdtf")
        self.assertEqual([call.args[0] for call in dependencies.call_args_list], ["faster_whisper"])
        runtime.assert_not_called()
        cookies.assert_not_called()
        for flags in (["--dataset-language", "Spanish"], ["--metadata", "other.json"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                pipeline.validate_args(parser, self.hdtf_args(*flags))


if __name__ == "__main__":
    unittest.main()
