"""Deterministic offline checks for bounded, ordered download and alignment pools."""

import contextlib
import io
import json
import threading
from pathlib import Path
from unittest.mock import Mock, patch

import train
import train_downvid as pipeline
from shofo_download import ShofoAccessError, ShofoClip
from tests.test_train_downvid import PipelineFixture


class ParallelDownloadTests(PipelineFixture):
    def setUp(self):
        super().setUp()
        self.rows = [{"id": str(i), "info": {"Language": "English"},
                      "clip": ShofoClip("a" * 40, f"videos/{i}.mp4", 5)} for i in range(10)]
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.enterContext(contextlib.redirect_stderr(io.StringIO()))
        self.enterContext(patch.object(pipeline, "iter_shofo", side_effect=lambda args: (row for row in self.rows)))

    def args(self, *extra):
        return super().args("--dataset", "shofo", *extra)

    def video(self, clip):
        output = self.root / Path(clip.filename).name
        if not output.exists():
            output.write_bytes(b"video")
        return output

    @contextlib.contextmanager
    def alignment_fixture(self, callback=None):
        grouper = Mock()
        tokenizer = pipeline.CharacterAlignmentTokenizer()

        def align(video, tokenizer, output, **kwargs):
            if callback:
                callback(video, tokenizer, **kwargs)
            manifest = {"text": "hi", "windows": [[0, 1]]}
            pipeline.write_json(output / "manifest.json", manifest)
            return manifest

        grouper.group_frames_by_token.side_effect = align
        with patch.object(pipeline, "load_grouper", return_value=grouper), \
                patch.object(pipeline, "alignment_tokenizer", return_value=(tokenizer, "character", {}, None)), \
                patch.object(pipeline, "training_record", side_effect=lambda manifest, tok, unit, video:
                             {**manifest, "video": str(video)}), \
                patch.object(pipeline, "download_shofo_clip", side_effect=lambda clip, args:
                             self.video(clip)) as download:
            yield grouper, tokenizer, download

    def records(self, args):
        return train.load_manifest(args.manifest, video_only=True)

    def test_parallel_completion_preserves_order_and_stops_at_exact_n(self):
        args = self.args("-N", "5", "--download-workers", "2")
        barrier = threading.Barrier(2)
        second_finished = threading.Event()
        lock = threading.Lock()
        calls, finished = [], []
        active = peak = 0

        def download(clip, args):
            nonlocal active, peak
            with lock:
                calls.append(clip.filename)
                active += 1
                peak = max(peak, active)
            if clip.filename in ("videos/0.mp4", "videos/1.mp4"):
                barrier.wait(timeout=5)
                if clip.filename == "videos/0.mp4":
                    self.assertTrue(second_finished.wait(5))
            output = self.video(clip)
            with lock:
                finished.append(clip.filename)
                active -= 1
            if clip.filename == "videos/1.mp4":
                second_finished.set()
            return output

        with patch.object(pipeline, "download_shofo_clip", side_effect=download):
            pipeline.prepare(args, video_only=True)
        self.assertEqual(peak, 2)
        self.assertEqual(finished[0], "videos/1.mp4")
        self.assertCountEqual(calls, [row["clip"].filename for row in self.rows[:5]])
        self.assertEqual([row["source_key"] for row in self.records(args)],
                         [row["clip"].source_key for row in self.rows[:5]])
        report = pipeline.read_cache(self.root / "pretrain_report.json")
        self.assertEqual((report["attempts"], report["usable"], report["download_workers"]), (5, 5, 2))

    def test_alignment_overlaps_downloads_and_stays_on_coordinator(self):
        args = self.args("-N", "2", "--download-workers", "2", "--sequential-alignment",
                         "--alignment-workers", "8")
        alignment_started, second_downloaded = threading.Event(), threading.Event()
        coordinator = threading.get_ident()
        aligned = []

        def download(clip, args):
            self.assertNotEqual(threading.get_ident(), coordinator)
            if clip.filename == "videos/1.mp4":
                self.assertTrue(alignment_started.wait(5))
                second_downloaded.set()
            return self.video(clip)

        def align(video, *args, **kwargs):
            self.assertEqual(threading.get_ident(), coordinator)
            aligned.append(video.name)
            alignment_started.set()
            self.assertTrue(second_downloaded.wait(5))
            return {"text": "hi", "windows": [[0, 1]]}

        grouper = Mock()
        grouper.group_frames_by_token.side_effect = align
        with patch.object(pipeline, "load_grouper", return_value=grouper), \
                patch.object(pipeline, "alignment_tokenizer", return_value=(None, "token", {}, None)), \
                patch.object(pipeline, "training_record", side_effect=lambda manifest, tok, unit, video:
                             {**manifest, "video": str(video)}), \
                patch.object(pipeline, "download_shofo_clip", side_effect=download):
            pipeline.prepare(args)
        self.assertEqual(aligned, ["0.mp4", "1.mp4"])
        grouper.WhisperModel.assert_called_once()
        self.assertNotIn("num_workers", grouper.WhisperModel.call_args.kwargs)
        self.assertEqual(pipeline.read_cache(self.root / "run_report.json")["alignment_workers"], 1)

    def test_parallel_alignment_shares_model_isolates_tokenizers_and_preserves_order(self):
        args = self.args("-N", "5", "--alignment-workers", "2", "--download-workers", "4")
        barrier = threading.Barrier(2)
        second_finished = threading.Event()
        coordinator = threading.get_ident()
        tokenizers, models, finished = {}, [], []
        lock = threading.Lock()
        active = peak = 0

        def align(video, tokenizer, **kwargs):
            nonlocal active, peak
            worker = threading.get_ident()
            self.assertNotEqual(worker, coordinator)
            with lock:
                active += 1
                peak = max(peak, active)
                if worker in tokenizers:
                    self.assertIs(tokenizers[worker], tokenizer)
                tokenizers[worker] = tokenizer
                models.append(kwargs["whisper_instance"])
            if video.name in ("0.mp4", "1.mp4"):
                barrier.wait(timeout=5)
                if video.name == "0.mp4":
                    self.assertTrue(second_finished.wait(5))
            with lock:
                finished.append(video.name)
                active -= 1
            if video.name == "1.mp4":
                second_finished.set()

        with self.alignment_fixture(align) as (grouper, original, download):
            pipeline.prepare(args)
        self.assertEqual(peak, 2)
        self.assertEqual(finished[0], "1.mp4")
        self.assertEqual(len({id(tokenizer) for tokenizer in tokenizers.values()}), 2)
        self.assertTrue(all(tokenizer is not original for tokenizer in tokenizers.values()))
        self.assertTrue(all(model is models[0] for model in models))
        grouper.WhisperModel.assert_called_once_with(args.whisper_model, device=args.whisper_device,
                                                    compute_type=args.whisper_compute_type, num_workers=2)
        self.assertEqual(download.call_count, 5)
        self.assertEqual([row["source_key"] for row in self.records(args)],
                         [row["clip"].source_key for row in self.rows[:5]])
        report = pipeline.read_cache(self.root / "run_report.json")
        self.assertEqual((report["alignment_workers"], report["attempts"], report["usable"]), (2, 5, 5))

    def test_parallel_alignment_with_serial_downloads(self):
        args = self.args("-N", "2", "--download-workers", "1")
        barrier = threading.Barrier(2)
        with self.alignment_fixture(lambda *args, **kwargs: barrier.wait(timeout=5)) as (_, _, download):
            pipeline.prepare(args)
        self.assertEqual(download.call_count, 2)
        self.assertEqual(len(self.records(args)), 2)

    def test_parallel_fast_tokenizers_produce_valid_training_records(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        backend = Tokenizer(models.WordLevel({"<unk>": 0, "hello": 1, "world": 2}, unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
        args = self.args("-N", "2")
        barrier = threading.Barrier(2)
        grouper = Mock()

        def align(video, local_tokenizer, output, **kwargs):
            self.assertIsNot(local_tokenizer, tokenizer)
            barrier.wait(timeout=5)
            encoded = local_tokenizer("hello world", add_special_tokens=False, return_offsets_mapping=True)
            manifest = {"text": "hello world", "frames_decoded": 2, "groups": [[0, 0], [1, 1]],
                        "tokens": [{"token_id": token, "char_start": offset[0], "char_end": offset[1]}
                                   for token, offset in zip(encoded["input_ids"], encoded["offset_mapping"])]}
            pipeline.write_json(output / "manifest.json", manifest)
            return manifest

        grouper.group_frames_by_token.side_effect = align
        with patch.object(pipeline, "load_grouper", return_value=grouper), \
                patch.object(pipeline, "alignment_tokenizer", return_value=(tokenizer, "token", {}, None)), \
                patch.object(pipeline, "download_shofo_clip", side_effect=lambda clip, args: self.video(clip)):
            pipeline.prepare(args)
        records = train.load_manifest(args.manifest)
        self.assertEqual(len(records), 2)
        self.assertTrue(all(row["text"] == "hello world" and row["window_unit"] == "token" for row in records))

    def test_alignment_cache_reused_across_worker_counts_and_reprocessed_on_request(self):
        args = self.args("-N", "3", "--alignment-workers", "2")
        with self.alignment_fixture() as (grouper, _, _):
            pipeline.prepare(args)
            previous = args.manifest.read_bytes()
            grouper.reset_mock()
            args.sequential_alignment = True
            pipeline.prepare(args)
            grouper.WhisperModel.assert_not_called()
            grouper.group_frames_by_token.assert_not_called()
            self.assertEqual(args.manifest.read_bytes(), previous)
            args.reprocess = True
            pipeline.prepare(args)
            self.assertEqual(grouper.group_frames_by_token.call_count, 3)
            grouper.WhisperModel.assert_called_once()

    def test_alignment_failure_replaced_without_exceeding_remaining_clip_budget(self):
        args = self.args("-N", "2", "--alignment-workers", "4", "--max-attempts", "3")

        def align(video, tokenizer, **kwargs):
            if video.name == "0.mp4":
                raise ValueError("unusable speech alignment")

        with self.alignment_fixture(align) as (_, _, download):
            pipeline.prepare(args)
        self.assertEqual(download.call_count, 3)
        self.assertEqual([row["source_key"] for row in self.records(args)], ["tiktok:1", "tiktok:2"])
        report = pipeline.read_cache(self.root / "run_report.json")
        self.assertEqual((report["attempts"], report["usable"], len(report["failures"])), (3, 2, 1))
        self.assertFalse(list((self.root / "frames" / self.rows[0]["clip"].key).glob("*/complete.json")))

    def test_parallel_model_setup_failure_is_not_retried_or_committed(self):
        args = self.args("-N", "4")
        args.manifest.write_text("previous manifest\n")
        with self.alignment_fixture() as (grouper, _, download):
            grouper.WhisperModel.side_effect = RuntimeError("unavailable device")
            with self.assertRaisesRegex(pipeline.AlignmentSetupError, "Cannot initialize Whisper"):
                pipeline.prepare(args)
            grouper.WhisperModel.assert_called_once()
            self.assertLessEqual(download.call_count, args.num_videos)
        self.assertEqual(args.manifest.read_text(), "previous manifest\n")
        self.assertFalse(list(self.root.glob("frames/**/complete.json")))
        report = pipeline.read_cache(self.root / "run_report.json")
        self.assertEqual(report["usable"], 0)
        self.assertIn("unavailable device", report["failures"][0]["error"])

    def test_prefetch_respects_selection_deduplication_and_exclusions(self):
        args = self.args("-N", "2", "--download-workers", "4", "--start-index", "1",
                         "--dataset-language", "en")
        excluded = self.root / "excluded.jsonl"
        excluded.write_text(json.dumps({"video": str(self.root / "other.mp4"),
                                        "source_key": self.rows[1]["clip"].source_key}) + "\n")
        args.exclude_manifest = excluded
        self.rows[2]["info"]["Language"] = "Spanish"
        self.rows.insert(4, self.rows[3])
        calls = []

        def download(clip, args):
            calls.append(clip.filename)
            return self.video(clip)

        with patch.object(pipeline, "download_shofo_clip", side_effect=download):
            pipeline.prepare(args, video_only=True)
        self.assertCountEqual(calls, ["videos/3.mp4", "videos/4.mp4"])
        report = pipeline.read_cache(self.root / "pretrain_report.json")
        self.assertEqual((report["skipped_excluded_clips"], report["attempts"]), (1, 2))

    def test_failures_use_attempt_budget_and_keep_previous_manifest(self):
        args = self.args("-N", "3", "--download-workers", "4", "--max-attempts", "4")
        args.manifest.write_text("previous manifest\n")
        calls = []

        def download(clip, args):
            calls.append(clip.filename)
            if clip.filename in ("videos/0.mp4", "videos/1.mp4"):
                raise OSError("invalid video")
            return self.video(clip)

        with patch.object(pipeline, "download_shofo_clip", side_effect=download), \
                self.assertRaisesRegex(RuntimeError, "Only 2/3 usable clips after 4 attempts"):
            pipeline.prepare(args, video_only=True)
        self.assertCountEqual(calls, [row["clip"].filename for row in self.rows[:4]])
        self.assertEqual(args.manifest.read_text(), "previous manifest\n")
        partial = train.load_manifest(str(args.manifest) + ".partial", video_only=True)
        self.assertEqual(len(partial), 2)

    def test_fatal_error_stops_prefetch_and_joins_active_workers(self):
        args = self.args("-N", "6", "--download-workers", "2")
        barrier = threading.Barrier(2)
        finished = threading.Event()
        calls = []

        def download(clip, args):
            calls.append(clip.filename)
            barrier.wait(timeout=5)
            if clip.filename == "videos/0.mp4":
                raise ShofoAccessError("access denied")
            output = self.video(clip)
            finished.set()
            return output

        with patch.object(pipeline, "download_shofo_clip", side_effect=download), \
                self.assertRaisesRegex(ShofoAccessError, "access denied"):
            pipeline.prepare(args, video_only=True)
        self.assertTrue(finished.is_set())
        self.assertCountEqual(calls, ["videos/0.mp4", "videos/1.mp4"])
        self.assertFalse(args.manifest.exists())
        report = pipeline.read_cache(self.root / "pretrain_report.json")
        self.assertEqual((report["attempts"], report["usable"]), (2, 0))

    def test_serial_mode_does_not_prefetch_on_fatal_error(self):
        args = self.args("-N", "6", "--download-workers", "1")
        with patch.object(pipeline, "download_shofo_clip", side_effect=ShofoAccessError("denied")) as download, \
                self.assertRaises(ShofoAccessError):
            pipeline.prepare(args, video_only=True)
        download.assert_called_once()

    def test_worker_count_defaults_and_validation(self):
        self.assertEqual(self.args().download_workers, 4)
        self.assertEqual(self.args().alignment_workers, 4)
        self.assertFalse(self.args().sequential_alignment)
        for flag in ("--download-workers", "--alignment-workers"):
            for value in ("0", "-1", "1.5"):
                with self.subTest(flag=flag, value=value), self.assertRaises(SystemExit):
                    self.args(flag, value)
