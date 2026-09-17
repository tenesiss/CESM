"""Offline coverage of TalkVid selection, alignment, caching, and trainer handoff."""

import contextlib
import importlib.util
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import PropertyMock, patch

import cv2
import numpy as np
import torch

import train
import train_downvid as pipeline


class PipelineFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def args(self, *extra):
        args = pipeline.build_argparser().parse_args([
            "-N", "1", "--work-dir", str(self.root), *extra,
        ])
        args.manifest = self.root / "data.jsonl"
        args.group_frames_code = pipeline.DEFAULT_GROUPER
        args.max_attempts = args.max_attempts or 10 * args.num_videos
        return args

    def row(self, number=0, **extra):
        return {"id": f"clip-{number}", "start-time": 0, "end-time": 1,
                "info": {"Video Link": f"https://www.youtube.com/watch?v=test{number}",
                         "Language": "English"}, **extra}


class PipelineTests(PipelineFixture):

    def test_download_command_cut_audio_and_cache(self):
        args = self.args("--js-runtimes", "node:/runtime path/node")
        clip = pipeline.Clip.from_row(self.row())
        calls = []

        def download(command, **kwargs):
            calls.append(command)
            template = command[command.index("--output") + 1]
            Path(template.replace("%(ext)s", "mp4")).write_bytes(b"fake downloaded video")

        with patch.object(pipeline, "run_download_process", side_effect=download), \
                patch.object(pipeline, "probe_video") as probe:
            result = pipeline.download_clip(clip, args)
            self.assertEqual(pipeline.download_clip(clip, args), result)
        self.assertEqual(len(calls), 1)
        self.assertIn("*0.000000000-1.000000000", calls[0])
        self.assertIn("--force-keyframes-at-cuts", calls[0])
        self.assertNotIn("--extract-audio", calls[0])
        self.assertIn("ffmpeg_i:-rw_timeout 30000000 -short_seek_size 1", calls[0])
        self.assertEqual(calls[0][calls[0].index("--js-runtimes") + 1], "node:/runtime path/node")
        probe.assert_called_once()
        self.assertTrue(result.is_file())


    def test_download_timeout_does_not_cache_incomplete_video(self):
        args = self.args()
        clip = pipeline.Clip.from_row(self.row())
        with patch.object(pipeline, "run_download_process",
                          side_effect=subprocess.TimeoutExpired("yt-dlp", 600)):
            with self.assertRaisesRegex(RuntimeError, "timed out after 600s"):
                pipeline.download_clip(clip, args)
        self.assertIsNone(pipeline.cached_download(clip, args))
        self.assertFalse((pipeline.download_folder(clip, args) / "download.json").exists())


    @unittest.skipUnless(os.name == "nt", "Windows process-tree regression")
    def test_timeout_stops_real_windows_download_child(self):
        pid_file = self.root / "child.pid"
        child = "import time; time.sleep(60)"
        parent = (
            "import subprocess, sys; from pathlib import Path; "
            f"p = subprocess.Popen([sys.executable, '-c', {child!r}]); "
            f"Path({str(pid_file)!r}).write_text(str(p.pid)); p.wait()"
        )
        try:
            with (self.root / "process.log").open("w") as log:
                with self.assertRaises(subprocess.TimeoutExpired):
                    pipeline.run_download_process([sys.executable, "-c", parent], stdout=log, timeout=3)
            self.assertTrue(pid_file.is_file(), "Test child did not start")
            pid = pid_file.read_text()
            result = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                                    capture_output=True, text=True, check=True, timeout=10)
            self.assertNotIn(f'"{pid}"', result.stdout)
        finally:
            if pid_file.exists():
                subprocess.run(["taskkill", "/PID", pid_file.read_text(), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is unavailable")
class DownloadFrameLimitIntegrationTests(PipelineFixture):

    def test_real_ffmpeg_limits_frames_and_audio_at_clip_start(self):
        # Serve a local fixture over HTTP so the real downloader's network
        # options apply, without depending on YouTube or internet access.
        from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
        from threading import Thread

        source = self.root / "source.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
            "testsrc2=size=32x32:rate=25:duration=4", "-f", "lavfi", "-i",
            "sine=frequency=440:duration=4", "-c:v", "libx264", "-c:a", "aac", str(source),
        ], check=True, timeout=30)
        class Handler(SimpleHTTPRequestHandler):
            def __init__(handler, *args, **kwargs):
                super().__init__(*args, directory=str(self.root), **kwargs)

            def log_message(handler, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        Thread(target=server.serve_forever, daemon=True).start()
        source_url = f"http://127.0.0.1:{server.server_port}/{source.name}"
        run_process = subprocess.run

        def download(command, **kwargs):
            if "yt_dlp" not in command:
                return run_process(command, **kwargs)
            start, end = map(float, command[command.index("--download-sections") + 1][1:].split("-"))
            options = command[command.index("--downloader-args") + 1]
            self.assertTrue(options.startswith("ffmpeg_o:"))
            self.assertNotIn("-fps_mode", options)
            self.assertIn("-vsync 0", options)
            output = command[command.index("--output") + 1].replace("%(ext)s", "mp4")
            input_options = next(value for value in command if value.startswith("ffmpeg_i:"))
            return run_process([
                "ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(end - start),
                *shlex.split(input_options.split(":", 1)[1]),
                "-i", source_url, *shlex.split(options.split(":", 1)[1]), output,
            ], stderr=subprocess.STDOUT, check=True, **kwargs)

        for start, end, limit, expected_frames in (
            (0, 4, 12, 12), (1, 4, 12, 12), (1, 2, 50, 25), (1, 4, 1, 1),
        ):
            with self.subTest(start=start, end=end, limit=limit):
                args = self.args("--download-max-frames", str(limit))
                clip = pipeline.Clip.from_row(self.row(**{"start-time": start, "end-time": end}))
                with patch.object(pipeline, "run_download_process", side_effect=download):
                    output = pipeline.download_clip(clip, args)
                result = run_process([
                    "ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(output),
                ], capture_output=True, text=True, check=True, timeout=30)
                streams = {stream["codec_type"]: stream for stream in json.loads(result.stdout)["streams"]}
                self.assertEqual(int(streams["video"]["nb_read_frames"]), expected_frames)
                self.assertAlmostEqual(float(streams["video"]["duration"]), expected_frames / 25, places=2)
                self.assertAlmostEqual(float(streams["audio"]["duration"]), expected_frames / 25, delta=0.05)


@unittest.skipUnless(pipeline.DEFAULT_GROUPER.is_file()
                     and importlib.util.find_spec("faster_whisper"), "local frame grouper is unavailable")
class GrouperIntegrationTests(PipelineFixture):

    def setUp(self):
        super().setUp()
        self.grouper = pipeline.load_grouper(pipeline.DEFAULT_GROUPER)
        self.video = self.root / "fixture.avi"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"MJPG"), 12, (32, 32))
        self.assertTrue(writer.isOpened())
        for index in range(12):
            writer.write(np.full((32, 32, 3), index * 15, dtype=np.uint8))
        writer.release()
        self.words = [self.grouper.WordSpan("hi", 0, 2, 0.1, 0.8)]


    def test_grouped_frames_support_unicode_folders_in_all_image_formats(self):
        for ext in ("jpg", "jpeg", "png"):
            with self.subTest(ext=ext):
                output = self.root / "r\u00e9sultats_\u4e2d" / ext
                spans = [
                    self.grouper.TokenSpan(0, 1, "\u0120This", "This", 0, 4, 0, 0.4),
                    self.grouper.TokenSpan(1, 2, "\u2581works", " works", 4, 10, 0.4, 0.8),
                ]
                manifest = self.grouper.save_grouped_frames(
                    self.video, spans, output, transcript="This works",
                    image_ext=ext, jpeg_quality=37,
                )
                self.assertEqual(manifest["groups"], [[0, 4], [5, 9]])
                self.assertEqual(manifest["frames_decoded"], 12)
                self.assertEqual(manifest["frames_assigned_to_tokens"], 10)
                for token, (start, end) in zip(manifest["tokens"], manifest["groups"]):
                    folder = output / token["folder"]
                    self.assertIn(token["token"], folder.name)
                    files = sorted(folder.glob(f"*.{ext}"))
                    self.assertEqual([file.name for file in files],
                                     [f"frame_{i:09d}.{ext}" for i in range(start, end + 1)])
                    for index, file in enumerate(files, start):
                        # Decode bytes through Python too: imread has the same
                        # Unicode filename limitation as imwrite on Windows.
                        frame = cv2.imdecode(np.frombuffer(file.read_bytes(), dtype=np.uint8),
                                             cv2.IMREAD_COLOR)
                        self.assertIsNotNone(frame)
                        self.assertEqual(frame.shape, (32, 32, 3))
                        self.assertLessEqual(np.abs(frame.astype(int) - index * 15).max(), 3)
                self.assertEqual(pipeline.read_cache(output / "manifest.json"), manifest)


    def test_frame_write_error_includes_path_and_releases_video(self):
        output = self.root / "write-failure"
        spans = [self.grouper.TokenSpan(0, 1, "This", "This", 0, 4, 0, 0.8)]
        cap = cv2.VideoCapture(str(self.video))
        self.addCleanup(cap.release)
        with patch.object(self.grouper.cv2, "VideoCapture", return_value=cap), \
                patch.object(Path, "write_bytes", side_effect=PermissionError(13, "Permission denied")):
            with self.assertRaisesRegex(RuntimeError, "Failed to write frame:.*Permission denied") as caught:
                self.grouper.save_grouped_frames(self.video, spans, output)
        self.assertIn("frame_000000000.jpg", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, PermissionError)
        self.assertFalse(cap.isOpened())
        self.assertFalse((output / "manifest.json").exists())



if __name__ == "__main__":
    unittest.main()
