"""Offline Shofo access, selective downloads, media validation, and pipeline routing."""

import contextlib
import io
import json
import shutil
import subprocess
import unittest
from dataclasses import replace
from http.client import IncompleteRead
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request
from unittest.mock import Mock, patch

import huggingface_hub

import shofo_download as shofo
import train
import train_downvid as pipeline
from tests.test_train_downvid import PipelineFixture


def response(payload):
    stream = io.BytesIO(payload)
    stream.status = 200
    return stream


class ShofoAccessTests(PipelineFixture):
    def setUp(self):
        super().setUp()
        self.enterContext(patch.object(huggingface_hub, "get_token", return_value="test-secret"))
        self.clip = shofo.ShofoClip("a" * 40, "videos/12/12345.mp4", 5)

    def test_listing_selects_only_mp4s_and_pins_commit(self):
        info = SimpleNamespace(sha="a" * 40, siblings=[
            SimpleNamespace(rfilename=name, size=5)
            for name in ("metadata.parquet", "preview.mp4", "videos/34/34567.mp4", "videos/12/12345.mp4")
        ])
        args = self.args("--dataset", "shofo", "--shofo-revision", "release")
        with patch.object(huggingface_hub.HfApi, "dataset_info", return_value=info) as listing:
            rows = list(shofo.iter_shofo(args))
        self.assertEqual([row["id"] for row in rows], ["12345", "34567"])
        self.assertEqual(rows[0]["clip"], self.clip)
        listing.assert_called_once_with(shofo.SHOFO_REPO, revision="release", files_metadata=True,
                                        timeout=args.download_timeout)
        newer = replace(self.clip, revision="b" * 40)
        self.assertNotEqual(self.clip.key, newer.key)
        self.assertEqual(self.clip.source_key, newer.source_key)
        self.assertEqual(self.clip.source_key, "tiktok:12345")

    def test_listing_access_failure_is_actionable_and_not_retried(self):
        error = RuntimeError("signed URL with test-secret")
        error.response = SimpleNamespace(status_code=403)
        with patch.object(huggingface_hub.HfApi, "dataset_info", side_effect=error) as listing:
            with self.assertRaisesRegex(shofo.ShofoAccessError, "hf auth login") as caught:
                list(shofo.iter_shofo(self.args("--dataset", "shofo")))
        self.assertNotIn("test-secret", str(caught.exception))
        self.assertEqual(listing.call_count, 1)

    def test_listing_transient_failure_retries_and_invalid_listing_stops(self):
        error = RuntimeError("service unavailable")
        error.response = SimpleNamespace(status_code=503)
        info = SimpleNamespace(sha="a" * 40, siblings=[
            SimpleNamespace(rfilename=self.clip.filename, size=5),
        ])
        with patch.object(huggingface_hub.HfApi, "dataset_info", side_effect=[error, info]), \
                patch.object(shofo.time, "sleep"):
            self.assertEqual(len(list(shofo.iter_shofo(self.args()))), 1)
        for filename, size in (("videos/../bad.mp4", 5), (self.clip.filename, None),
                               (self.clip.filename, 0), ("metadata.parquet", 5)):
            info.siblings = [SimpleNamespace(rfilename=filename, size=size)]
            with self.subTest(filename=filename, size=size), \
                    patch.object(huggingface_hub.HfApi, "dataset_info", return_value=info), \
                    self.assertRaises(shofo.ShofoAccessError):
                list(shofo.iter_shofo(self.args()))

    def test_download_auth_pinned_url_and_retry_replaces_partial_bytes(self):
        opener = Mock()
        interrupted = response(b"bad")
        opener.open.side_effect = [interrupted, response(b"video")]
        output = self.root / "source.mp4"
        with patch.object(shofo, "build_opener", return_value=opener), patch.object(shofo.time, "sleep"):
            shofo.copy_shofo_video(self.clip, output, 30, 1)
        self.assertEqual(output.read_bytes(), b"video")
        self.assertEqual(opener.open.call_count, 2)
        request = opener.open.call_args.args[0]
        self.assertIn(f"/resolve/{self.clip.revision}/{self.clip.filename}", request.full_url)
        self.assertEqual(request.get_header("Authorization"), "Bearer test-secret")

    def test_redirect_keeps_token_on_same_origin_only(self):
        handler = shofo.HubRedirectHandler()
        request = Request("https://huggingface.co/start", headers={"Authorization": "Bearer test-secret"})
        for target, expected in (("https://huggingface.co/end", "Bearer test-secret"),
                                 ("https://cdn.example/video?signature=secret", None),
                                 ("http://huggingface.co/end", None)):
            redirected = handler.redirect_request(request, None, 302, "Found", {}, target)
            self.assertEqual(redirected.get_header("Authorization"), expected)

    def test_transfer_errors_do_not_expose_credentials(self):
        for status, exception in ((401, shofo.ShofoAccessError), (403, shofo.ShofoAccessError),
                                   (404, OSError)):
            opener = Mock()
            opener.open.side_effect = HTTPError("https://cdn.example?test-secret", status,
                                                 "test-secret", {}, None)
            with self.subTest(status=status), patch.object(shofo, "build_opener", return_value=opener):
                with self.assertRaises(exception) as caught:
                    shofo.copy_shofo_video(self.clip, self.root / "source.mp4", 30, 3)
                self.assertEqual(opener.open.call_count, 1)
                self.assertNotIn("test-secret", str(caught.exception))

    def test_retry_http_and_interrupted_read_and_deadline(self):
        interrupted = response(b"")
        interrupted.read1 = Mock(side_effect=IncompleteRead(b"partial", 5))
        opener = Mock()
        opener.open.side_effect = [HTTPError("https://example.test", 503, "retry", {}, None),
                                   interrupted, response(b"video")]
        with patch.object(shofo, "build_opener", return_value=opener), patch.object(shofo.time, "sleep"):
            shofo.copy_shofo_video(self.clip, self.root / "source.mp4", 30, 2)
        self.assertEqual(opener.open.call_count, 3)
        with patch.object(shofo, "build_opener", return_value=opener), \
                patch.object(shofo.time, "monotonic", side_effect=[0, 31]):
            with self.assertRaises(TimeoutError):
                shofo.copy_shofo_video(self.clip, self.root / "source.mp4", 30, 3)

    def test_cli_requires_only_hosted_dependencies(self):
        parser = pipeline.build_argparser()
        args = parser.parse_args(["--dataset", "shofo", "-N", "1", "--dataset-language", "en"])
        with patch.object(pipeline.importlib.util, "find_spec", return_value=object()) as dependencies, \
                patch.object(pipeline.shutil, "which", return_value="binary"), \
                patch.object(pipeline, "validate_download_runtimes") as runtime, \
                patch.object(pipeline, "use_saved_login") as cookies:
            pipeline.validate_args(parser, args)
        self.assertEqual(args.work_dir, pipeline.ROOT / "data" / "shofo")
        self.assertEqual([call.args[0] for call in dependencies.call_args_list],
                         ["faster_whisper", "huggingface_hub"])
        runtime.assert_not_called()
        cookies.assert_not_called()
        for flags in (["--dataset-language", "Spanish"], ["--metadata", "other.json"],
                      ["--shofo-revision", ""]):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                pipeline.validate_args(parser, self.args("--dataset", "shofo", *flags))


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class ShofoPipelineTests(PipelineFixture):
    def setUp(self):
        super().setUp()
        self.source = self.root / "source.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
            "testsrc2=size=32x32:rate=25:duration=2", "-f", "lavfi", "-i",
            "sine=frequency=440:duration=2", "-c:v", "libx264", "-c:a", "aac", str(self.source),
        ], check=True, timeout=30)
        self.payloads = {"videos/00/000.mp4": b"broken", **{
            f"videos/12/12{i}.mp4": self.source.read_bytes() for i in range(3)
        }}
        info = SimpleNamespace(sha="a" * 40, siblings=[
            SimpleNamespace(rfilename=name, size=len(payload)) for name, payload in self.payloads.items()
        ])
        self.enterContext(patch.object(huggingface_hub.HfApi, "dataset_info", return_value=info))
        self.enterContext(patch.object(huggingface_hub, "get_token", return_value="test-secret"))
        self.opener = Mock()
        self.opener.open.side_effect = lambda request, **kw: response(
            self.payloads[request.full_url.split(f"/resolve/{info.sha}/")[1]])
        self.enterContext(patch.object(shofo, "build_opener", return_value=self.opener))
        self.enterContext(patch.object(pipeline, "download_clip", side_effect=AssertionError("YouTube called")))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.enterContext(contextlib.redirect_stderr(io.StringIO()))

    def shofo_args(self, *flags):
        return self.args("--dataset", "shofo", "--dataset-language", "en", *flags)

    def test_frame_caps_audio_and_recovery_cache(self):
        clip = list(shofo.iter_shofo(self.shofo_args()))[1]["clip"]
        paths = []
        for limit, expected in ((None, 50), (12, 12), (1, 1), (100, 50)):
            args = self.shofo_args(*([] if limit is None else ["--download-max-frames", str(limit)]))
            video = pipeline.download_shofo_clip(clip, args)
            calls = self.opener.open.call_count
            self.assertEqual(pipeline.download_shofo_clip(clip, args), video)
            self.assertEqual(self.opener.open.call_count, calls)
            result = subprocess.run([
                "ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(video),
            ], check=True, capture_output=True, text=True, timeout=30)
            streams = {s["codec_type"]: s for s in json.loads(result.stdout)["streams"]}
            self.assertEqual(int(streams["video"]["nb_read_frames"]), expected)
            self.assertAlmostEqual(float(streams["audio"]["duration"]), expected / 25, delta=.06)
            paths.append(video)
        self.assertEqual(len(set(paths)), 4)
        paths[-1].write_bytes(b"corrupt cached file")
        pipeline.download_shofo_clip(clip, args)
        self.assertGreater(paths[-1].stat().st_size, len(b"corrupt cached file"))

    def test_exact_n_alignment_cache_and_failed_candidate(self):
        args = self.shofo_args("-N", "2", "--download-max-frames", "12")
        grouper = pipeline.load_grouper(pipeline.DEFAULT_GROUPER)
        words = [grouper.WordSpan("hi", 0, 2, .04, .3)]
        with patch.object(pipeline, "load_grouper", return_value=grouper), \
                patch.object(grouper, "WhisperModel"), \
                patch.object(grouper, "transcribe_words", return_value=("hi", words)) as asr:
            pipeline.prepare(args)
            self.assertEqual(asr.call_count, 2)
            pipeline.prepare(args)
            self.assertEqual(asr.call_count, 2)
        rows = train.load_manifest(args.manifest)
        self.assertEqual([row["source_key"] for row in rows], ["tiktok:120", "tiktok:121"])
        report = pipeline.read_cache(self.root / "run_report.json")
        self.assertEqual((report["dataset"], report["usable"], report["attempts"]), ("shofo", 2, 3))
        self.assertEqual(len(report["failures"]), 1)
        broken = list(shofo.iter_shofo(args))[0]["clip"]
        self.assertIsNone(pipeline.cached_download(broken, args))
        self.assertFalse(list(pipeline.download_folder(broken, args).glob("download-*")))

    def test_pretraining_exclusion_start_index_and_shortfall(self):
        args = self.shofo_args("-N", "2", "--start-index", "1", "--download-max-frames", "3")
        excluded = self.root / "excluded.jsonl"
        excluded.write_text(json.dumps({"video": str(self.source), "source_key": "tiktok:120"}) + "\n")
        args.exclude_manifest = excluded
        with patch.object(pipeline, "load_grouper", side_effect=AssertionError("Whisper loaded")):
            pipeline.prepare(args, video_only=True)
        rows = train.load_manifest(args.manifest, video_only=True)
        self.assertEqual([row["source_key"] for row in rows], ["tiktok:121", "tiktok:122"])
        report = pipeline.read_cache(self.root / "pretrain_report.json")
        self.assertEqual((report["skipped_excluded_clips"], report["attempts"]), (1, 2))
        previous = args.manifest.read_bytes()
        args.start_index = 3
        with self.assertRaisesRegex(RuntimeError, "Only 1/2"):
            pipeline.prepare(args, video_only=True)
        self.assertEqual(args.manifest.read_bytes(), previous)
        self.assertEqual(len(train.load_manifest(str(args.manifest) + ".partial", video_only=True)), 1)

    def test_access_failure_stops_without_caching_or_replacing_manifest(self):
        args = self.shofo_args("-N", "2", "--start-index", "1")
        args.manifest.write_text("previous manifest\n")
        self.opener.open.side_effect = HTTPError("https://example.test?test-secret", 403, "denied", {}, None)
        with self.assertRaises(shofo.ShofoAccessError):
            pipeline.prepare(args, video_only=True)
        self.assertEqual(self.opener.open.call_count, 1)
        self.assertEqual(args.manifest.read_text(), "previous manifest\n")
        report = pipeline.read_cache(self.root / "pretrain_report.json")
        self.assertEqual((report["usable"], report["attempts"]), (0, 1))
        self.assertNotIn("test-secret", json.dumps(report))
        self.assertFalse(list(self.root.glob("clips/**/download.json")))


if __name__ == "__main__":
    unittest.main()
