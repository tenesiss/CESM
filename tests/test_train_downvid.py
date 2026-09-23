"""Offline coverage of TalkVid selection, alignment, caching, and trainer handoff."""

import contextlib
import importlib.util
import io
import json
import os
import shlex
import shutil
import subprocess
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
    def test_all_training_arguments_round_trip(self):
        args = self.args("--amp", "--flash-mono", "--max-frames", "12",
                         "--download-max-frames", "50",
                         "--js-runtimes", "node:/runtime path/node",
                         "--epochs", "2", "--device", "cpu", "--confidence-epochs", "3",
                         "--pretrained-lm", "a/model with spaces", "--fine-tune-pretrained-lm",
                         "--output", "a path/checkpoint.pt", "--lambda-align", "0.2",
                         "--no-vproj", "--vcross-weight", "0.25")
        parsed = train.build_argparser().parse_args(pipeline.training_command(args)[2:])
        expected = {key: getattr(args, key) for key in vars(parsed)}
        expected["manifest"] = str(args.manifest)
        self.assertEqual(vars(parsed), expected)
        options = lambda p: {flag for action in p._actions for flag in action.option_strings}
        self.assertLessEqual(options(train.build_argparser()), options(pipeline.build_argparser()))

    def test_download_limit_must_be_positive_and_defaults_to_full_clip(self):
        self.assertIsNone(self.args().download_max_frames)
        for limit in ("0", "-1", "1.5"):
            with self.subTest(limit=limit), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.args("--download-max-frames", limit)

    def test_zero_start_valid_and_invalid_times_rejected(self):
        clip = pipeline.Clip.from_row(self.row())
        self.assertEqual(clip.start, 0)
        for start, end in [(0, 0), (-1, 3), (float("nan"), 2), (0, float("inf"))]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                pipeline.Clip.from_row(self.row(**{"start-time": start, "end-time": end}))

    @unittest.skipUnless(importlib.util.find_spec("ijson"), "install requirements-talkvid.txt")
    def test_stream_json_array_jsonl_and_bom(self):
        rows = [self.row(), self.row(1)]
        for content in [json.dumps(rows), "\n".join(map(json.dumps, rows))]:
            path = self.root / "metadata.json"
            path.write_text("\ufeff \n" + content, encoding="utf-8")
            self.assertEqual(list(pipeline.iter_metadata(str(path))), rows)

    def test_character_offsets_and_explicit_unit(self):
        tokenizer = pipeline.CharacterAlignmentTokenizer()
        text = " é🙂"
        encoded = tokenizer(text)
        manifest = {"text": text, "frames_decoded": 12, "groups": [[0, 3], [4, 7], [8, 11]],
                    "tokens": [{"token_id": token, "char_start": i, "char_end": i + 1}
                               for i, token in enumerate(encoded["input_ids"])]}
        record = pipeline.training_record(manifest, tokenizer, "character", self.root / "v.mp4")
        self.assertEqual(record["window_unit"], "character")
        self.assertEqual(record["windows"], manifest["groups"])
        manifest["tokens"].pop()
        with self.assertRaisesRegex(ValueError, "dropped tokens"):
            pipeline.training_record(manifest, tokenizer, "character", self.root / "v.mp4")

    def test_empty_windows_checked_with_trainer_repair(self):
        tokenizer = pipeline.CharacterAlignmentTokenizer()
        manifest = {"text": "ab", "frames_decoded": 3, "groups": [[0, -1], [0, 2]],
                    "tokens": [{"token_id": ord(ch), "char_start": i, "char_end": i + 1}
                               for i, ch in enumerate("ab")]}
        pipeline.training_record(manifest, tokenizer, "character", self.root / "v.mp4")
        manifest["groups"] = [[0, -1], [0, 0]]
        with self.assertRaisesRegex(ValueError, "too few"):
            pipeline.training_record(manifest, tokenizer, "character", self.root / "v.mp4")

    def test_pretrained_and_resume_use_identical_tokenizer(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        backend = Tokenizer(models.WordLevel({"<unk>": 0, "hello": 1, "world": 2}, unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = train.HuggingFaceTokenizer(
            PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>"), "test-model",
        )
        args = self.args("--pretrained-lm", "test-model")
        with patch.object(train.HuggingFaceTokenizer, "from_pretrained", return_value=tokenizer):
            fresh, unit, state, _ = pipeline.alignment_tokenizer(args)
        self.assertEqual(unit, "token")
        checkpoint = self.root / "resume.pt"
        torch.save({"tokenizer": state,
                    "model_config": {"pretrained_lm_name_or_path": "test-model"}}, checkpoint)
        args.resume = str(checkpoint)
        resumed, unit, _, _ = pipeline.alignment_tokenizer(args)
        self.assertEqual(fresh("hello world"), resumed("hello world"))
        args.pretrained_lm = "different-model"
        with self.assertRaisesRegex(ValueError, "does not match"):
            pipeline.alignment_tokenizer(args)

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
        self.assertNotIn("--downloader", calls[0])
        self.assertEqual(calls[0][calls[0].index("--js-runtimes") + 1], "node:/runtime path/node")
        probe.assert_called_once()
        self.assertTrue(result.is_file())

    def test_runtime_autodetection_and_override(self):
        installed = {"node": "/usr/bin/node", "nodejs": "/usr/bin/nodejs", "qjs": "/opt/qjs"}
        with patch.object(pipeline.shutil, "which", side_effect=installed.get):
            self.assertEqual(pipeline.download_runtimes(self.args()),
                             ["node:/usr/bin/node", "quickjs:/opt/qjs"])
            self.assertEqual(pipeline.download_runtimes(self.args("--js-runtimes", "deno:/custom/deno")),
                             ["deno:/custom/deno"])
        with patch.object(pipeline.shutil, "which", return_value=None):
            self.assertEqual(pipeline.download_runtimes(self.args()), [])

    def test_runtime_validation_rejects_node20_and_accepts_node22_or_deno(self):
        for runtime, version, valid in (("node", "v20.19.0", False),
                                         ("node", "v22.0.0", True),
                                         ("deno", "deno 2.2.0", False),
                                         ("deno", "deno 2.3.0", True)):
            with self.subTest(runtime=runtime, version=version):
                args = self.args("--js-runtimes", f"{runtime}:/runtime/{runtime}")
                result = subprocess.CompletedProcess([], 0, version + "\n", "")
                with patch.object(pipeline.subprocess, "run", return_value=result) as run, \
                        contextlib.redirect_stdout(io.StringIO()):
                    if valid:
                        pipeline.validate_download_runtimes(args)
                        self.assertEqual(args.js_runtimes, [f"{runtime}:{Path('/runtime') / runtime}"])
                    else:
                        with self.assertRaisesRegex(pipeline.DownloadSetupError, version.removeprefix("v")):
                            pipeline.validate_download_runtimes(args)
                self.assertEqual(run.call_args.kwargs["timeout"], 10)

    def test_runtime_validation_keeps_supported_alternative_and_rejects_missing(self):
        args = self.args("--js-runtimes", "node:/runtime/node", "--js-runtimes", "deno:/runtime/deno")
        results = [subprocess.CompletedProcess([], 0, "v20.19.0\n", ""),
                   subprocess.CompletedProcess([], 0, "deno 2.3.0\n", "")]
        with patch.object(pipeline.subprocess, "run", side_effect=results), \
                contextlib.redirect_stdout(io.StringIO()):
            pipeline.validate_download_runtimes(args)
        self.assertEqual(args.js_runtimes, [f"deno:{Path('/runtime/deno')}"])
        with patch.object(pipeline.shutil, "which", return_value=None):
            with self.assertRaisesRegex(pipeline.DownloadSetupError, "none on PATH"):
                pipeline.validate_download_runtimes(self.args())

    def test_unsupported_runtime_stops_main_before_preparation(self):
        with patch.object(pipeline, "validate_download_runtimes",
                          side_effect=pipeline.DownloadSetupError("node 20.19.0 is unsupported")) as check, \
                patch.object(pipeline.importlib.util, "find_spec", return_value=object()), \
                patch.object(pipeline.shutil, "which", return_value="/available"), \
                patch.object(pipeline, "prepare") as prepare, \
                patch.object(pipeline.subprocess, "run") as run:
            with self.assertRaisesRegex(pipeline.DownloadSetupError, "node 20.19.0"):
                pipeline.main(["-N", "5", "--work-dir", str(self.root),
                               "--group-frames-code", str(Path(pipeline.__file__))])
        check.assert_called_once()
        prepare.assert_not_called()
        run.assert_not_called()

    def test_source_identity_handles_youtube_aliases(self):
        urls = ["https://www.youtube.com/watch?v=--a-J9CP1NE&t=4",
                "https://youtu.be/--a-J9CP1NE?si=tracking", "https://m.youtube.com/shorts/--a-J9CP1NE"]
        for url in urls:
            self.assertEqual(pipeline.Clip(url, 0, 1).source_key, "youtube:--a-J9CP1NE")
        self.assertNotEqual(pipeline.Clip(urls[0], 0, 1).source_key,
                            pipeline.Clip("https://youtu.be/--gjwFLQqlI", 0, 1).source_key)

    def test_download_errors_expose_reason_and_classify_source_or_setup(self):
        args = self.args()
        clip = pipeline.Clip.from_row(self.row())
        for message, error_type in (
            ("ERROR: [youtube] test: Video unavailable", pipeline.SourceUnavailableError),
            ("ERROR: [youtube] test: This video is unavailable", pipeline.SourceUnavailableError),
            ("ERROR: [youtube] test: Private video. Sign in if you've been granted access", pipeline.SourceUnavailableError),
            ("ERROR: [youtube] test: Sign in to confirm you’re not a bot", pipeline.DownloadSetupError),
            ("ERROR: HTTP Error 429: Too Many Requests", pipeline.DownloadSetupError),
            ("ERROR: [youtube] test: The page needs to be reloaded.", pipeline.DownloadSetupError),
            ("yt-dlp: error: no such option: --js-runtimes", pipeline.DownloadSetupError),
            ("ERROR: Could not find chrome cookies database", pipeline.DownloadSetupError),
            ("ERROR: [youtube] test: Requested format is not available", RuntimeError),
            ("ERROR: unable to download video data: HTTP Error 403: Forbidden", RuntimeError),
        ):
            with self.subTest(message=message):
                def fail(command, **kwargs):
                    kwargs["stdout"].write(message + "\n")
                    raise subprocess.CalledProcessError(1, command)

                with patch.object(pipeline, "run_download_process", side_effect=fail):
                    with self.assertRaises(error_type) as caught:
                        pipeline.download_clip(clip, args)
                self.assertIs(type(caught.exception), error_type)
                self.assertIn(message, str(caught.exception))
                self.assertIn("download.log", str(caught.exception))

    def test_ffmpeg_option_error_exposes_underlying_reason_and_stops_run(self):
        log = self.root / "download.log"
        log.write_text(
            "WARNING: [youtube] No supported JavaScript runtime could be found.\n"
            "Unrecognized option 'fps_mode'.\n"
            "Error splitting the argument list: Option not found\n"
            "ERROR: ffmpeg exited with code 1\n",
            encoding="utf-8",
        )
        error = pipeline.download_process_error(log, 1)
        self.assertIs(type(error), pipeline.DownloadSetupError)
        self.assertIn("Unrecognized option 'fps_mode'", str(error))
        self.assertIn(str(log), str(error))

        # Other FFmpeg failures can be specific to the clip and remain skippable.
        log.write_text("Invalid data found when processing input\nERROR: ffmpeg exited with code 1\n",
                       encoding="utf-8")
        self.assertIs(type(pipeline.download_process_error(log, 1)), RuntimeError)

    def test_download_limits_use_separate_caches_and_validate_frame_count(self):
        clip = pipeline.Clip.from_row(self.row())
        paths = []

        def download(command, **kwargs):
            template = command[command.index("--output") + 1]
            Path(template.replace("%(ext)s", "mp4")).write_bytes(b"fake downloaded video")

        with patch.object(pipeline, "run_download_process", side_effect=download) as run, \
                patch.object(pipeline, "probe_video") as probe:
            for limit in (None, 12, 24):
                args = self.args(*([] if limit is None else ["--download-max-frames", str(limit)]))
                path = pipeline.download_clip(clip, args)
                paths.append(path)
                self.assertEqual(pipeline.download_clip(clip, args), path)
                self.assertEqual(probe.call_args.kwargs["max_frames"], limit)
                self.assertEqual(pipeline.read_cache(path.parent / "download.json")["max_frames"], limit)
            self.assertEqual(run.call_count, 3)
        self.assertEqual(len(set(paths)), 3)
        self.assertTrue(all(path.exists() for path in paths))

    def test_probe_accepts_capped_video_but_rejects_excess_frames_and_early_eof(self):
        for count, duration, limit, expected_error in (
            (12, 0.48, 12, None),  # Capped before the metadata section ends.
            (250, 10, 300, None),  # Source section is shorter than the cap.
            (13, 0.52, 12, "expected 1..12"),
            (0, 0, 12, "expected 1..12"),
            ("N/A", 0.48, 12, "Cannot count"),
            (11, 0.44, 12, "differs from requested"),  # Premature EOF.
            (12, 10, 12, "differs from requested"),  # Untrimmed audio tail.
        ):
            with self.subTest(count=count, duration=duration, limit=limit):
                result = subprocess.CompletedProcess([], 0, json.dumps({
                    "streams": [{"codec_type": "video", "nb_read_frames": str(count),
                                 "duration": "0.48"}, {"codec_type": "audio"}],
                    "format": {"duration": str(duration)},
                }))
                with patch.object(pipeline.subprocess, "run", return_value=result) as run:
                    if expected_error:
                        with self.assertRaisesRegex(ValueError, expected_error):
                            pipeline.probe_video(self.root / "clip.mp4", 10, max_frames=limit)
                    else:
                        pipeline.probe_video(self.root / "clip.mp4", 10, max_frames=limit)
                    self.assertIn("-count_frames", run.call_args.args[0])

    def test_no_audio_is_rejected(self):
        result = subprocess.CompletedProcess([], 0, json.dumps({
            "streams": [{"codec_type": "video"}], "format": {"duration": "1"},
        }))
        with patch.object(pipeline.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(ValueError, "both video and audio"):
                pipeline.probe_video(self.root / "silent.mp4", 1)

    def test_failed_prepare_never_launches_training(self):
        with patch.object(pipeline, "validate_args"), \
                patch.object(pipeline.subprocess, "run", side_effect=subprocess.CalledProcessError(1, [])) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                pipeline.main(["-N", "2"])
        self.assertEqual(run.call_count, 1)
        self.assertIn("--prepare-only", run.call_args.args[0])


@unittest.skipUnless(importlib.util.find_spec("yt_dlp"), "yt-dlp is unavailable")
class LoginTests(PipelineFixture):
    def cookie_file(self, rows=None):
        path = self.root / "cookies.txt"
        rows = rows if rows is not None else [
            ".youtube.com\tTRUE\t/\tTRUE\t0\tSAPISID\tfake-session-secret",
            ".youtube.com\tTRUE\t/\tTRUE\t1\tSID\texpired-secret",
            ".example.com\tTRUE\t/\tTRUE\t0\tSID\tunrelated-secret",
        ]
        path.write_text("# Netscape HTTP Cookie File\n" + "\n".join(rows) + "\n", encoding="utf-8")
        return path

    def test_cookie_import_is_private_filtered_and_reused(self):
        from yt_dlp.cookies import YoutubeDLCookieJar

        source = self.cookie_file()
        args = self.args("--login", "--cookies", str(source))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            pipeline.login(args)
        saved = pipeline.login_cookie_path(args)
        jar = YoutubeDLCookieJar(str(saved))
        jar.load()
        self.assertEqual([(c.domain, c.name) for c in jar], [(".youtube.com", "SAPISID")])
        self.assertNotIn("secret", output.getvalue())
        if os.name == "posix":
            self.assertEqual(saved.stat().st_mode & 0o777, 0o600)
        fresh = self.args()
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.use_saved_login(fresh)
        self.assertEqual(fresh.cookies, saved)
        explicit = self.args("--cookies", str(source))
        pipeline.use_saved_login(explicit)
        self.assertEqual(explicit.cookies, source)
        browser = self.args("--cookies-from-browser", "firefox")
        pipeline.use_saved_login(browser)
        self.assertIsNone(browser.cookies)

    def test_browser_import_forwards_profile_without_saving_other_sites(self):
        import yt_dlp
        from yt_dlp.cookies import YoutubeDLCookieJar

        jar = YoutubeDLCookieJar(str(self.cookie_file()))
        jar.load()
        args = self.args("--login", "--cookies-from-browser", "chrome:Profile 1")
        with patch.object(yt_dlp.YoutubeDL, "cookiejar", new_callable=PropertyMock, return_value=jar), \
                patch.object(yt_dlp, "parse_options", wraps=yt_dlp.parse_options) as parse, \
                contextlib.redirect_stdout(io.StringIO()):
            pipeline.login(args)
        self.assertEqual(parse.call_args.args[0],
                         ["--ignore-config", "--cookies-from-browser", "chrome:Profile 1"])
        saved = pipeline.login_cookie_path(args).read_text()
        self.assertIn(".youtube.com", saved)
        self.assertNotIn("example.com", saved)

    def test_invalid_expired_and_unsigned_cookies_preserve_previous_session(self):
        args = self.args()
        saved = pipeline.login_cookie_path(args)
        saved.parent.mkdir(parents=True)
        saved.write_text("previous saved session", encoding="utf-8")
        for rows in (
            [".youtube.com\tTRUE\t/\tTRUE\t0\tVISITOR_INFO1_LIVE\tvisitor-secret"],
            [".youtube.com\tTRUE\t/\tTRUE\t1\tSAPISID\texpired-secret"],
            ['[{"value": "malformed-secret"}]'],
        ):
            with self.subTest(rows=rows):
                args.cookies = self.cookie_file(rows)
                output = io.StringIO()
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                    with self.assertRaises(RuntimeError) as caught:
                        pipeline.login(args)
                self.assertNotIn("secret", str(caught.exception) + output.getvalue())
                self.assertEqual(saved.read_text(), "previous saved session")

    def test_login_and_logout_exit_without_dataset_preparation(self):
        source = self.cookie_file()
        flags = ["--work-dir", str(self.root)]
        with patch.object(pipeline, "validate_args") as validate, \
                patch.object(pipeline, "prepare") as prepare, \
                patch.object(pipeline.subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()):
            pipeline.main(["--login", "--cookies", str(source), *flags])
            self.assertTrue(pipeline.login_cookie_path(self.args()).is_file())
            pipeline.main(["--logout", *flags])
            self.assertFalse(pipeline.login_cookie_path(self.args()).exists())
            self.assertTrue(source.exists())
            validate.assert_not_called()
            prepare.assert_not_called()
            run.assert_not_called()

    def test_normal_runs_still_require_n(self):
        parser = pipeline.build_argparser()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            pipeline.validate_args(parser, parser.parse_args([]))

    def test_interactive_login_prompts_for_path_or_browser(self):
        for text, attribute, value in (("/tmp/cookies.txt", "cookies", Path("/tmp/cookies.txt")),
                                       ("browser:chrome:Profile 1", "cookies_from_browser", "chrome:Profile 1")):
            with self.subTest(text=text):
                args = self.args()
                with patch.object(pipeline.sys.stdin, "isatty", return_value=True), \
                        patch("builtins.input", return_value=text), contextlib.redirect_stdout(io.StringIO()):
                    pipeline.prompt_login_source(args)
                self.assertEqual(getattr(args, attribute), value)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is unavailable")
class DownloadFrameLimitIntegrationTests(PipelineFixture):
    def test_real_ffmpeg_limits_frames_and_audio_at_clip_start(self):
        # Replace only URL extraction: execute the pipeline's actual FFmpeg
        # options against a local source, then run its real ffprobe validation.
        source = self.root / "source.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
            "testsrc2=size=32x32:rate=25:duration=4", "-f", "lavfi", "-i",
            "sine=frequency=440:duration=4", "-c:v", "libx264", "-c:a", "aac", str(source),
        ], check=True, timeout=30)
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
            return run_process([
                "ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(end - start),
                "-i", str(source), *shlex.split(options.split(":", 1)[1]), output,
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


class PretrainingPreparationTests(PipelineFixture):
    def test_positive_count_and_video_only_validation_skip_alignment_dependencies(self):
        for count in ("0", "-1", "1.5"):
            with self.subTest(count=count), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args("--N-pretrain", count)
        parser = pipeline.build_argparser()
        args = parser.parse_args([
            "--N-pretrain", "3", "--prepare-only", "--work-dir", str(self.root),
            "--group-frames-code", str(self.root / "missing.py"),
        ])
        checked = []

        def dependency(name):
            checked.append(name)
            return None if name == "faster_whisper" else object()

        with patch.object(pipeline.importlib.util, "find_spec", side_effect=dependency), \
                patch.object(pipeline, "validate_download_runtimes"), patch.object(pipeline.shutil, "which", return_value="tool"):
            pipeline.validate_args(parser, args)
        self.assertNotIn("faster_whisper", checked)
        self.assertTrue(args.pretrain_visual_encoder)
        self.assertEqual(args.pretrain_manifest, str(self.root / "pretrain.jsonl"))
        self.assertEqual(args.max_attempts, 30)

    def test_pretraining_replaces_failed_clips_excludes_validation_and_skips_alignment(self):
        args = self.args("-N", "2")
        args.manifest = self.root / "pretrain.jsonl"
        excluded_clip = pipeline.Clip.from_row(self.row(1))
        args.validation_manifest = str(self.root / "valid.jsonl")
        Path(args.validation_manifest).write_text(json.dumps({"video": "heldout.mp4",
                                                            "source_key": excluded_clip.source_key}) + "\n")
        attempted = []

        def download(clip, args):
            attempted.append(clip.url)
            if clip.url.endswith("test0"):
                raise RuntimeError("broken clip")
            path = self.root / f"{clip.key}.mp4"
            path.write_bytes(b"downloaded fixture")
            return path

        rows = [self.row(), self.row(1), self.row(2), self.row(2), self.row(3)]
        with patch.object(pipeline, "iter_metadata", return_value=(row for row in rows)), \
                patch.object(pipeline, "download_clip", side_effect=download), \
                patch.object(pipeline, "load_grouper", side_effect=AssertionError("alignment was loaded")), \
                patch.object(pipeline, "alignment_tokenizer", side_effect=AssertionError("tokenizer was loaded")), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            pipeline.prepare(args, video_only=True)
        records = train.load_manifest(args.manifest, video_only=True)
        self.assertEqual(len(records), 2)
        self.assertTrue(all(set(row) == {"video", "source_key"} for row in records))
        self.assertEqual(attempted, [pipeline.Clip.from_row(self.row(i)).url for i in (0, 2, 3)])
        report = pipeline.read_cache(self.root / "pretrain_report.json")
        self.assertEqual((report["requested"], report["usable"], report["attempts"]), (2, 2, 3))
        self.assertEqual(report["skipped_excluded_clips"], 1)

    def test_pretraining_shortfall_preserves_previous_manifest(self):
        args = self.args("-N", "2")
        args.manifest.write_text("previous manifest\n")
        video = self.root / "fixture.mp4"
        video.write_bytes(b"fixture")
        with patch.object(pipeline, "iter_metadata", return_value=(self.row() for _ in range(1))), \
                patch.object(pipeline, "download_clip", return_value=video), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "Only 1/2 usable"):
                pipeline.prepare(args, video_only=True)
        self.assertEqual(args.manifest.read_text(), "previous manifest\n")
        self.assertEqual(len(train.load_manifest(str(args.manifest) + ".partial", video_only=True)), 1)

    def test_combined_downloads_forward_pretraining_manifest_and_options(self):
        preparations, commands = [], []

        def prepare(args, *, video_only=False):
            preparations.append((args.num_videos, str(args.manifest), video_only))

        def run(command, **kwargs):
            commands.append(command)
            if "--prepare-only" in command:
                pipeline.main(command[2:])

        with patch.object(pipeline.importlib.util, "find_spec", return_value=object()), \
                patch.object(pipeline, "validate_download_runtimes"), \
                patch.object(pipeline.shutil, "which", return_value="tool"), \
                patch.object(pipeline, "prepare", side_effect=prepare), \
                patch.object(pipeline.subprocess, "run", side_effect=run), contextlib.redirect_stdout(io.StringIO()):
            pipeline.main(["-N", "2", "--N-pretrain", "5", "--work-dir", str(self.root),
                           "--pretrain-epochs", "3", "--pretrain-adjacent-frames", "4"])
        self.assertEqual(preparations, [(2, str(self.root / "data.jsonl"), False),
                                        (5, str(self.root / "pretrain.jsonl"), True)])
        forwarded = train.build_argparser().parse_args(commands[-1][2:])
        self.assertTrue(forwarded.pretrain_visual_encoder)
        self.assertEqual(forwarded.pretrain_manifest, str(self.root / "pretrain.jsonl"))
        self.assertEqual((forwarded.pretrain_epochs, forwarded.pretrain_adjacent_frames), (3, 4))
        self.assertFalse(any(flag.startswith("--N-pretrain") for flag in commands[-1]))


@unittest.skipUnless(pipeline.DEFAULT_GROUPER.is_file()
                     and importlib.util.find_spec("faster_whisper"), "local frame grouper is unavailable")
class GrouperIntegrationTests(PipelineFixture):
    """Exercise the local grouper and CESM data loader with synthetic clips."""

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

    def prepare_with_fixtures(self, args, rows, downloader=None):
        with patch.object(pipeline, "iter_metadata", return_value=(row for row in rows)), \
                patch.object(pipeline, "load_grouper", return_value=self.grouper), \
                patch.object(pipeline, "download_clip", side_effect=downloader or (lambda clip, args: self.video)), \
                patch.object(self.grouper, "WhisperModel"), \
                patch.object(self.grouper, "transcribe_words", return_value=("hi", self.words)) as transcribe:
            pipeline.prepare(args)
        return transcribe.call_count

    def test_failed_and_duplicate_clips_replaced_cache_and_exact_n(self):
        args = self.args("-N", "2", "--dataset-language", "English")
        rows = [self.row(), self.row(), self.row(1), self.row(2), self.row(3)]
        calls = []

        def downloader(clip, args):
            calls.append(clip.url)
            if clip.url.endswith("test0"):
                raise RuntimeError("unavailable")
            return self.video

        self.assertEqual(self.prepare_with_fixtures(args, rows, downloader), 2)
        records = train.load_manifest(str(args.manifest))
        self.assertEqual(len(records), 2)
        self.assertEqual(len(calls), 3)
        tokenizer = train.CharTokenizer.build(row["text"] for row in records)
        dataset = train.VideoTextWindowDataset(records, tokenizer, 16, use_face_detector=False)
        self.assertEqual(dataset[0]["video"].shape[0], 12)
        self.assertEqual(self.prepare_with_fixtures(args, rows, downloader), 0)
        args.language = "en"
        self.assertEqual(self.prepare_with_fixtures(args, rows, downloader), 2)
        args.download_max_frames = 8
        self.assertEqual(self.prepare_with_fixtures(args, rows, downloader), 2)
        self.assertEqual(pipeline.read_cache(self.root / "run_report.json")["download_max_frames"], 8)
        args.download_max_frames = None
        self.assertEqual(self.prepare_with_fixtures(args, rows, downloader), 0)

    def test_shortfall_keeps_previous_manifest_and_partial_results(self):
        args = self.args("-N", "2", "--max-attempts", "2")
        args.manifest.write_text("previous dataset\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Only 1/2"):
            self.prepare_with_fixtures(args, [self.row()])
        self.assertEqual(args.manifest.read_text(), "previous dataset\n")
        self.assertEqual(len(train.load_manifest(str(args.manifest) + ".partial")), 1)

    def test_validation_excludes_known_uploads_and_legacy_paths_without_using_attempts(self):
        args = self.args("--max-attempts", "1")
        rows = [self.row(), self.row(**{"start-time": 2, "end-time": 3}), self.row(1), self.row(2)]
        excluded = self.root / "training.jsonl"
        excluded.write_text("\n".join(map(json.dumps, [
            {"video": str(self.root / "training.mp4"), "text": "hi", "windows": [[0, 1], [2, 3]],
             "source_key": pipeline.Clip.from_row(rows[0]).source_key},
            {"video": str(pipeline.download_folder(pipeline.Clip.from_row(rows[2]), args) / "video.mp4"),
             "text": "hi", "windows": [[0, 1], [2, 3]]},
        ])) + "\n")
        args.exclude_manifest = excluded
        calls = []

        def downloader(clip, args):
            calls.append(clip)
            return self.video

        self.prepare_with_fixtures(args, rows, downloader)
        self.assertEqual(calls, [pipeline.Clip.from_row(rows[3])])
        report = pipeline.read_cache(self.root / "run_report.json")
        self.assertEqual(report["skipped_excluded_clips"], 3)
        self.assertEqual(report["attempts"], 1)
        self.assertEqual(train.load_manifest(args.manifest)[0]["source_key"], calls[0].source_key)

    def test_dead_upload_does_not_use_attempts_for_its_remaining_clips(self):
        args = self.args("--max-attempts", "2")
        rows = [self.row(i) for i in range(60)]
        for i, row in enumerate(rows):
            row["start-time"], row["end-time"] = i, i + 1
            row["info"]["Video Link"] = ("https://www.youtube.com/watch?v=--a-J9CP1NE" if i % 2
                                          else "https://youtu.be/--a-J9CP1NE?si=tracking")
        rows.append(self.row(100))
        calls = []

        def downloader(clip, args):
            calls.append(clip.url)
            if clip.source_key == "youtube:--a-J9CP1NE":
                raise pipeline.SourceUnavailableError("Video unavailable")
            return self.video

        self.assertEqual(self.prepare_with_fixtures(args, rows, downloader), 1)
        self.assertEqual(len(calls), 2)
        report = pipeline.read_cache(self.root / "run_report.json")
        self.assertEqual(report["attempts"], 2)
        self.assertEqual(report["skipped_unavailable_clips"], 59)
        self.assertEqual(report["usable"], 1)

    def test_temporary_failure_keeps_other_clips_from_same_upload_eligible(self):
        args = self.args("--max-attempts", "2")
        rows = [self.row(), self.row(**{"start-time": 2, "end-time": 3})]
        calls = []

        def downloader(clip, args):
            calls.append(clip.url)
            if len(calls) == 1:
                raise RuntimeError("HTTP Error 403")
            return self.video

        self.assertEqual(self.prepare_with_fixtures(args, rows, downloader), 1)
        self.assertEqual(len(calls), 2)

    def test_unavailable_upload_still_allows_already_cached_clips(self):
        args = self.args("--max-attempts", "2")
        rows = [self.row(), self.row(**{"start-time": 2, "end-time": 3})]
        clip = pipeline.Clip.from_row(rows[1])
        folder = pipeline.download_folder(clip, args)
        folder.mkdir(parents=True)
        video = folder / "video.mp4"
        shutil.copyfile(self.video, video)
        pipeline.write_json(folder / "download.json", {"video_signature": pipeline.video_signature(video)})
        download_cached_clip = pipeline.download_clip

        def downloader(clip, args):
            if clip.start == 0:
                raise pipeline.SourceUnavailableError("Video unavailable")
            return download_cached_clip(clip, args)

        self.assertEqual(self.prepare_with_fixtures(args, rows, downloader), 1)
        self.assertEqual(pipeline.read_cache(self.root / "run_report.json")["usable"], 1)

    def test_setup_error_stops_immediately_and_is_reported(self):
        args = self.args("--max-attempts", "50")
        calls = []

        def downloader(clip, args):
            calls.append(clip.url)
            raise pipeline.DownloadSetupError("Sign in to confirm you're not a bot")

        with self.assertRaises(pipeline.DownloadSetupError):
            self.prepare_with_fixtures(args, [self.row(), self.row(1)], downloader)
        self.assertEqual(len(calls), 1)
        report = pipeline.read_cache(self.root / "run_report.json")
        self.assertEqual(report["attempts"], 1)
        self.assertIn("not a bot", report["failures"][0]["error"])

    def test_reload_error_stops_before_trying_another_clip_or_upload(self):
        args = self.args("--max-attempts", "50")
        log = self.root / "reload.log"
        log.write_text("ERROR: [youtube] test: The page needs to be reloaded.\n", encoding="utf-8")
        calls = []

        def downloader(clip, args):
            calls.append(clip.url)
            raise pipeline.download_process_error(log, 1)

        rows = [self.row(), self.row(**{"start-time": 2, "end-time": 3}), self.row(1)]
        with self.assertRaisesRegex(pipeline.DownloadSetupError, "update the downloader"):
            self.prepare_with_fixtures(args, rows, downloader)
        self.assertEqual(len(calls), 1)
        report = pipeline.read_cache(self.root / "run_report.json")
        self.assertEqual(report["attempts"], 1)
        self.assertEqual(report["unavailable_sources"], {})
        self.assertIn("page needs to be reloaded", report["failures"][0]["error"])

    def test_prepare_then_train_real_tiny_cpu_checkpoint(self):
        args = self.args("--epochs", "1", "--confidence-epochs", "1", "--batch-size", "1",
                         "--device", "cpu", "--mouth-size", "16", "--d-video", "8", "--d-text", "8",
                         "--d-fusion", "8", "--heads", "2", "--video-layers", "1", "--text-layers", "1",
                         "--conv3d-channels", "4", "--no-face-detector", "--max-frames", "8",
                         "--output", str(self.root / "model.pt"))
        self.prepare_with_fixtures(args, [self.row()])
        # Run the same train.py entry function used by its CLI, with only train
        # parser arguments. Keep this tiny integration test fast on large CPUs.
        parsed = train.build_argparser().parse_args(pipeline.training_command(args)[2:])
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            train.train(parsed)
        finally:
            torch.set_num_threads(threads)
        checkpoint = torch.load(args.output, map_location="cpu", weights_only=False)
        self.assertEqual(checkpoint["epoch"], 1)
        self.assertEqual(checkpoint["confidence_epoch"], 1)
        self.assertTrue(checkpoint["confidence"]["trained"])


if __name__ == "__main__":
    unittest.main()
