"""All-in-one command routing, exact-sample handoff, and offline training smoke test."""

import contextlib
import csv
import hashlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

import run_all_variants as runner
import train
import train_downvid


class AllVariantRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.parser = runner.build_argparser()

    def plan(self, flags):
        args = self.parser.parse_args(flags)
        return runner.build_commands(self.parser, args, self.root / "run")

    def test_default_plan_prepares_once_and_keeps_all_variant_defaults(self):
        _, snapshot, checkpoints, stages = self.plan(["-N", "3"])
        self.assertEqual([s["name"] for s in stages], ["download", *[
            name for index in range(1, 6) for name in (f"train_{index}", f"evaluate_{index}")
        ]])
        preparation = train_downvid.build_argparser().parse_args(stages[0]["command"][2:])
        self.assertTrue(preparation.prepare_only)
        self.assertEqual(preparation.num_videos, 3)
        self.assertEqual(len(set(checkpoints)), 5)
        for index, (module, stage) in enumerate(zip(runner.VARIANTS, stages[1::2]), 1):
            actual = module.build_argparser().parse_args(stage["command"][2:])
            defaults = module.build_argparser().parse_args(["--manifest", "unused"])
            for key, expected in vars(defaults).items():
                if key not in ("manifest", "output", "metrics_csv"):
                    self.assertEqual(getattr(actual, key), expected, (index, key))
            self.assertEqual(actual.manifest, str(snapshot))
            self.assertEqual(actual.metrics_csv, "")  # Evaluation runs in its own process.
            evaluation = stages[index * 2]
            evaluate = evaluation["command"]
            self.assertEqual(evaluate[evaluate.index("--manifest") + 1], str(snapshot))
            self.assertEqual(evaluate[evaluate.index("--checkpoints") + 1:evaluate.index("--output")],
                             [str(checkpoints[index - 1])])
            self.assertEqual(evaluation["csv"], str(checkpoints[index - 1].with_suffix(".metrics.csv")))
            self.assertEqual(evaluate[evaluate.index("--output") + 1], evaluation["csv"])
            self.assertEqual(evaluate[evaluate.index("--batch-size") + 1], "2")
            self.assertEqual(evaluate[evaluate.index("--device") + 1], "auto")
            self.assertNotIn("--max-frames", evaluate)  # Read each checkpoint's own section size.

    def test_selected_variants_train_and_evaluate_once_in_numeric_order(self):
        for source in (["-N", "3"], ["--manifest", "existing.jsonl"]):
            for selection, expected in (([5], [5]), ([5, 2, 5], [2, 5])):
                with self.subTest(source=source, selection=selection):
                    _, snapshot, checkpoints, stages = self.plan([
                        *source, "--variants", *map(str, selection),
                        "--variant-args", "5:--lr 0.001 --max-frames 7",
                    ])
                    prefix = ["download"] if source[0] == "-N" else []
                    self.assertEqual([s["name"] for s in stages], prefix + [
                        name for index in expected for name in (f"train_{index}", f"evaluate_{index}")
                    ])
                    self.assertEqual(len(checkpoints), len(expected))
                    for offset, index in enumerate(expected):
                        module = runner.VARIANTS[index - 1]
                        training, evaluation = stages[len(prefix) + 2 * offset:len(prefix) + 2 * offset + 2]
                        actual = module.build_argparser().parse_args(training["command"][2:])
                        self.assertEqual(actual.variant_name, module.DEFAULTS["variant_name"])
                        self.assertEqual(actual.manifest, str(snapshot))
                        self.assertEqual(Path(actual.output), checkpoints[offset])
                        self.assertEqual(checkpoints[offset].name, Path(module.DEFAULTS["output"]).name)
                        self.assertEqual(evaluation["csv"], str(checkpoints[offset].with_suffix(".metrics.csv")))
                        if index == 5:
                            self.assertEqual((actual.lr, actual.max_frames), (0.001, 7))

    def test_shared_specific_and_individual_parameters_route_to_correct_stages(self):
        _, _, _, stages = self.plan([
            "-N", "2", "--lr", "0.001", "--batch-size", "3", "--epochs", "2",
            "--lambda-tcross", "0.2", "--tcross-margin", "4", "--w-t", "0.5",
            "--lambda-align", "0.1", "--lambda-mono", "0.3", "--flash-mono",
            "--max-frames", "11", "--metrics-include-eos",
            "--pretrained-lm", "a/model with spaces", "--pretrained-lm-local-files-only",
            "--variant-args", "3:--lr 0.002 --batch-size 7 --max-frames 5",
            "--dataset-language", "English", "--dataset-language", "Spanish",
            "--download-max-frames", "27", "--cookies", str(self.root / "cookies file.txt"),
            "--eval-batch-size", "1", "--eval-device", "cpu",
        ])
        prepare = train_downvid.build_argparser().parse_args(stages[0]["command"][2:])
        self.assertEqual(prepare.dataset_language, ["English", "Spanish"])
        self.assertEqual(prepare.download_max_frames, 27)
        self.assertEqual(prepare.pretrained_lm, "a/model with spaces")
        self.assertTrue(prepare.pretrained_lm_local_files_only)
        for index, stage in enumerate(stages[1::2], 1):
            # Fusion defaults live in each script, so use that parser here.
            args = runner.VARIANTS[index - 1].build_argparser().parse_args(stage["command"][2:])
            self.assertEqual(args.lr, .002 if index == 3 else .001)
            self.assertEqual(args.batch_size, 7 if index == 3 else 3)
            self.assertEqual(args.max_frames, 5 if index == 3 else 11)
            self.assertEqual(args.lambda_tcross, .2 if index == 2 else 0)
            self.assertEqual(args.lambda_align, 0 if index == 5 else .1)
            self.assertEqual(args.lambda_mono, 0 if index == 5 else .3)
            self.assertEqual(args.flash_mono, index != 5)
            self.assertEqual(args.w_t, .5 if index == 4 else 1.)
            self.assertEqual(args.pretrained_lm, prepare.pretrained_lm)
        for stage in stages[2::2]:
            command = stage["command"]
            self.assertIn("--include-eos", command)
            self.assertEqual(command[command.index("--batch-size") + 1], "1")
            self.assertEqual(command[command.index("--device") + 1], "cpu")

    def test_hdtf_selection_and_frame_cap_route_only_to_downloader(self):
        prepared, _, _, stages = self.plan([
            "--dataset", "hdtf", "-N", "10", "--hdtf-archive", "archive with spaces.zip",
            "--start-index", "7", "--download-max-frames", "50", "--max-frames", "25",
            "--download-retries", "2", "--download-timeout", "90",
        ])
        self.assertEqual(prepared.name, "hdtf.jsonl")
        self.assertEqual(Path(stages[0]["command"][1]).name, "train_downvid.py")
        args = train_downvid.build_argparser().parse_args(stages[0]["command"][2:])
        self.assertEqual((args.dataset, args.num_videos, args.start_index), ("hdtf", 10, 7))
        self.assertEqual(args.hdtf_archive, "archive with spaces.zip")
        self.assertEqual(args.download_max_frames, 50)
        self.assertEqual((args.download_retries, args.download_timeout), (2, 90))
        self.assertIsNone(args.work_dir)  # Downloader chooses data/hdtf.
        for module, stage in zip(runner.VARIANTS, stages[1::2]):
            training = module.build_argparser().parse_args(stage["command"][2:])
            self.assertEqual(training.max_frames, 25)
            self.assertNotIn("--dataset=hdtf", stage["command"])

    def test_validation_download_and_reused_manifests_route_to_selected_variants(self):
        for source in (["-N", "3"], ["--manifest", "existing.jsonl"]):
            for validation in (["--N-valid", "2"], ["--validation-manifest", "validation.jsonl"]):
                with self.subTest(source=source, validation=validation):
                    _, snapshot, _, stages = self.plan([
                        *source, *validation, "--variants", "2", "5",
                        "--early-stopping-patience", "3", "--early-stopping-min-delta", "0.01",
                        "--variant-args", "5:--early-stopping-patience 7",
                    ])
                    downloads = [s for s in stages if s["name"].startswith("download")]
                    self.assertEqual(len(downloads), int(source[0] == "-N") + int(validation[0] == "--N-valid"))
                    if validation[0] == "--N-valid":
                        args = train_downvid.build_argparser().parse_args(downloads[-1]["command"][2:])
                        self.assertEqual(args.num_videos, 2)
                        self.assertEqual(args.exclude_manifest, snapshot)
                    elif source[0] == "-N":
                        args = train_downvid.build_argparser().parse_args(downloads[0]["command"][2:])
                        self.assertEqual(args.exclude_manifest, snapshot.with_name("validation_samples.jsonl"))
                    for stage in stages[len(downloads):]:
                        index = int(stage["name"].rsplit("_", 1)[1])
                        if stage["name"].startswith("train_"):
                            args = runner.VARIANTS[index - 1].build_argparser().parse_args(stage["command"][2:])
                            self.assertEqual(args.validation_manifest, str(self.root / "run" / "validation_samples.jsonl"))
                            self.assertEqual(args.early_stopping_patience, 7 if index == 5 else 3)
                            self.assertEqual(args.early_stopping_min_delta, 0.01)
                        else:
                            is_validation = stage["name"].startswith("evaluate_validation_")
                            expected = snapshot.with_name("validation_samples.jsonl") if is_validation else snapshot
                            command = stage["command"]
                            self.assertEqual(command[command.index("--manifest") + 1], str(expected))

    def test_invalid_or_dataset_changing_overrides_fail_before_download(self):
        for flags in (
            ["-N", "1", "--variant-args", "3:--manifest other.jsonl"],
            ["-N", "1", "--variant-args", "4:--pretrained-lm other"],
            ["-N", "1", "--variant-args", "5:--lambda-tcross 1"],
            ["-N", "1", "--variant-args", "1:--text-fusion frame"],
            ["-N", "1", "--variant-args", "1:--vcross-weight 0.25"],
            ["-N", "1", "--vcross-weight", "nan"],
            ["-N", "1", "--epochs", "0"],
            ["-N", "1", "--lr", "nan"],
            ["-N", "1", "--batch-size", "0"],
            ["-N", "1", "--manifest", "existing.jsonl"],
            ["-N", "1", "--variants"],
            ["-N", "1", "--variants", "0"],
            ["-N", "1", "--variants", "6"],
            ["-N", "1", "--variants", "baseline"],
            ["-N", "1", "--variants", "1", "--variant-args", "4:--lr 0.001"],
            ["-N", "1", "--N-valid", "0"],
            ["-N", "1", "--N-valid", "-1"],
            ["-N", "1", "--N-valid", "2", "--validation-manifest", "other.jsonl"],
            ["-N", "1", "--early-stopping-patience", "-1"],
            ["-N", "1", "--early-stopping-min-delta", "nan"],
            ["-N", "1", "--variant-args", "1:--validation-manifest other.jsonl"],
            ["-N", "1", "--N-pretrain", "0"],
            ["-N", "1", "--N-pretrain", "-1"],
            ["-N", "1", "--N-pretrain", "2", "--pretrain-manifest", "other.jsonl"],
            ["-N", "1", "--pretrain-visual-encoder"],
            ["-N", "1", "--N-pretrain", "2", "--pretrain-epochs", "0"],
            ["-N", "1", "--N-pretrain", "2", "--pretrain-adjacent-frames", "1"],
            ["-N", "1", "--lambda-pretrain-temporal", "-1"],
            ["-N", "1", "--lambda-pretrain-augmentation", "nan"],
            ["-N", "1", "--lambda-pretrain-variance", "inf"],
            ["-N", "1", "--lambda-pretrain-covariance", "-1"],
            ["-N", "1", "--pretrain-variance-floor", "0"],
            ["-N", "1", "--variant-args", "1:--pretrain-variance-floor nan"],
            ["-N", "1", "--variant-args", "1:--pretrain-manifest other.jsonl"],
        ):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.plan(flags)

    def test_pretraining_download_count_and_shared_manifest_route_separately(self):
        for dataset in ("talkvid", "hdtf"):
            with self.subTest(dataset=dataset):
                _, _, _, stages = self.plan([
                    "-N", "2", "--N-pretrain", "7", "--N-valid", "3", "--dataset", dataset,
                    "--variants", "1", "5", "--pretrain-epochs", "2", "--pretrain-adjacent-frames", "4",
                    "--pretrained-lm", "model", "--download-max-frames", "20",
                    "--lambda-pretrain-temporal", "0.2", "--lambda-pretrain-augmentation", "3",
                    "--lambda-pretrain-variance", "4", "--lambda-pretrain-covariance", "0.5",
                    "--pretrain-variance-floor", "0.8",
                    "--variant-args", "5:--lambda-pretrain-covariance 0.7 --pretrain-variance-floor 0.9",
                ])
                self.assertEqual([s["name"] for s in stages[:3]],
                                 ["download", "download_validation", "download_pretrain"])
                downloaded = [train_downvid.build_argparser().parse_args(s["command"][2:]) for s in stages[:3]]
                self.assertEqual([d.num_videos for d in downloaded], [2, 3, None])
                pretrain = downloaded[-1]
                self.assertEqual(pretrain.num_pretrain, 7)
                self.assertEqual(pretrain.dataset, dataset)
                self.assertEqual(pretrain.download_max_frames, 20)
                self.assertIsNone(pretrain.pretrained_lm)  # No tokenizer/alignment for unlabeled videos.
                self.assertEqual(pretrain.exclude_manifest, self.root / "run" / "validation_samples.jsonl")
                for stage in stages[3:]:
                    if stage["name"].startswith("train_"):
                        parsed = train.build_argparser().parse_args(stage["command"][2:])
                        self.assertTrue(parsed.pretrain_visual_encoder)
                        self.assertEqual(parsed.pretrain_manifest, str(self.root / "run" / "pretrain_samples.jsonl"))
                        self.assertEqual((parsed.pretrain_epochs, parsed.pretrain_adjacent_frames), (2, 4))
                        self.assertEqual((parsed.lambda_pretrain_temporal, parsed.lambda_pretrain_augmentation,
                                          parsed.lambda_pretrain_variance), (0.2, 3, 4))
                        expected = (0.7, 0.9) if stage["name"] == "train_5" else (0.5, 0.8)
                        self.assertEqual((parsed.lambda_pretrain_covariance, parsed.pretrain_variance_floor), expected)

    def test_reused_pretraining_manifest_is_snapshotted_and_excluded_from_validation(self):
        for name in ("train", "pretrain", "valid"):
            (self.root / f"{name}.mp4").write_bytes(b"fixture")
        training = {"video": str(self.root / "train.mp4"), "text": "a", "windows": [[0, 1]]}
        pretraining = {"video": str(self.root / "pretrain.mp4"), "source_key": "unlabeled-upload"}
        validation = dict(training, video=str(self.root / "valid.mp4"))
        manifest = self.root / "train.jsonl"
        manifest.write_text(json.dumps(training) + "\n")
        pretrain_manifest = self.root / "unlabeled.jsonl"
        pretrain_manifest.write_text(json.dumps(pretraining) + "\n")
        run_dir = self.root / "reuse"

        def stage(command, log):
            if log.stem == "download_validation":
                args = train_downvid.build_argparser().parse_args(command[2:])
                self.assertEqual(train.load_manifest(args.exclude_manifest, video_only=True),
                                 [training, pretraining])
                Path(args.manifest).write_text(json.dumps(validation) + "\n")
            elif log.stem == "train_1":
                args = train.build_argparser().parse_args(command[2:])
                self.assertTrue(args.pretrain_visual_encoder)
                self.assertEqual(train.load_manifest(args.pretrain_manifest, video_only=True), [pretraining])
            else:
                train.write_metrics_csv(command[command.index("--output") + 1], [])

        with patch.object(runner, "run_stage", side_effect=stage), contextlib.redirect_stdout(io.StringIO()):
            report = runner.main(["--manifest", str(manifest), "--pretrain-manifest", str(pretrain_manifest),
                                  "--N-valid", "1", "--variants", "1", "--run-dir", str(run_dir)])
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["pretrain_samples"]["count"], 1)
        self.assertEqual(report["pretrain_samples"]["manifest"], str(run_dir / "pretrain_samples.jsonl"))
        self.assertFalse(any(s["name"] == "download_pretrain" for s in report["stages"]))

    def test_short_pretraining_download_never_starts_training(self):
        video = self.root / "video.mp4"
        video.write_bytes(b"fixture")
        manifest = self.root / "train.jsonl"
        manifest.write_text(json.dumps({"video": str(video), "text": "a", "windows": [[0, 1]]}) + "\n")
        calls = []

        def stage(command, log):
            calls.append(log.stem)
            args = train_downvid.build_argparser().parse_args(command[2:])
            Path(args.pretrain_manifest).write_text(json.dumps({"video": str(video)}) + "\n")

        with patch.object(runner, "run_stage", side_effect=stage), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "expected exactly 2"):
                runner.main(["--manifest", str(manifest), "--N-pretrain", "2", "--variants", "1",
                             "--run-dir", str(self.root / "short")])
        self.assertEqual(calls, ["download_pretrain"])

    def test_projection_and_frame_weight_route_to_compatible_variants(self):
        _, _, _, stages = self.plan(['-N', '1', '--no-vproj', '--vcross-weight', '0.25'])
        for index, stage in enumerate(stages[1::2], 1):
            args = runner.VARIANTS[index - 1].build_argparser().parse_args(stage['command'][2:])
            self.assertTrue(args.no_vproj)
            self.assertEqual(args.vcross_weight, .25 if index == 5 else None)

    def test_snapshot_preserves_order_duplicates_windows_and_absolute_paths(self):
        (self.root / "clip.avi").write_bytes(b"fixture")
        rows = [{"video": "clip.avi", "text": text, "windows": [[0, 2]]} for text in ("a", "b")]
        source = self.root / "source.jsonl"
        source.write_text("\n".join(map(json.dumps, rows)) + "\n")
        snapshot = self.root / "snapshot.jsonl"
        result = runner.snapshot_samples(source, snapshot, 2)
        saved = train.load_manifest(snapshot)
        self.assertEqual([row["text"] for row in saved], ["a", "b"])
        self.assertEqual([row["windows"] for row in saved], [[[0, 2]], [[0, 2]]])
        self.assertTrue(all(row["video"] == str(self.root / "clip.avi") for row in saved))
        self.assertEqual(result["sha256"], hashlib.sha256(snapshot.read_bytes()).hexdigest())
        source.write_text("changed")
        runner.verify_snapshot(result)
        snapshot.write_text("changed")
        with self.assertRaisesRegex(RuntimeError, "manifest changed"):
            runner.verify_snapshot(result)

    def test_failed_download_or_wrong_count_never_starts_training(self):
        for mode in ("failure", "partial"):
            run_dir = self.root / mode
            calls = []

            def download(command, log):
                calls.append(command)
                if mode == "failure":
                    raise subprocess.CalledProcessError(1, command)
                args = train_downvid.build_argparser().parse_args(command[2:])
                Path(args.manifest).write_text(json.dumps({"video": "unused", "text": "a", "windows": [[0, 1]]}) + "\n")

            with contextlib.redirect_stdout(io.StringIO()), patch.object(runner, "run_stage", side_effect=download):
                with self.assertRaises((ValueError, subprocess.CalledProcessError)):
                    runner.main(["-N", "2", "--run-dir", str(run_dir)])
            self.assertEqual(len(calls), 1)
            report = json.loads((run_dir / "run.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertTrue(all(s["status"] == "pending" for s in report["stages"][1:]))

    def test_train_failure_never_evaluates_and_keeps_run_report(self):
        source = self.root / "source.jsonl"
        video = self.root / "clip.avi"
        video.write_bytes(b"fixture")
        source.write_text(json.dumps({"video": str(video), "text": "a", "windows": [[0, 1]]}) + "\n")
        calls = []

        def stage(command, log):
            calls.append(command)
            raise subprocess.CalledProcessError(4, command)

        run_dir = self.root / "failed_training"
        with contextlib.redirect_stdout(io.StringIO()), patch.object(runner, "run_stage", side_effect=stage):
            with self.assertRaises(subprocess.CalledProcessError):
                runner.main(["--manifest", str(source), "--run-dir", str(run_dir)])
        report = json.loads((run_dir / "run.json").read_text())
        self.assertEqual(len(calls), 1)
        self.assertEqual(report["stages"][0]["status"], "failed")
        self.assertTrue(all(s["status"] == "pending" for s in report["stages"][1:]))
        self.assertFalse((run_dir / "all_variants.csv").exists())

    def test_incomplete_or_overlapping_validation_never_starts_training(self):
        video = self.root / "clip.avi"
        video.write_bytes(b"fixture")
        row = {"video": str(video), "text": "a", "windows": [[0, 1]]}
        source = self.root / "training.jsonl"
        source.write_text(json.dumps(row) + "\n")
        for mode in ("incomplete", "overlap"):
            with self.subTest(mode=mode):
                run_dir = self.root / mode

                def download(command, log):
                    self.assertEqual(log.stem, "download_validation")
                    args = train_downvid.build_argparser().parse_args(command[2:])
                    count = 1 if mode == "incomplete" else 2
                    Path(args.manifest).write_text((json.dumps(row) + "\n") * count)

                with patch.object(runner, "run_stage", side_effect=download) as execute, \
                        contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(ValueError, "expected exactly|overlap"):
                        runner.main(["--manifest", str(source), "--N-valid", "2", "--run-dir", str(run_dir)])
                self.assertEqual(execute.call_count, 1)
                report = json.loads((run_dir / "run.json").read_text())
                self.assertEqual(report["status"], "failed")
                self.assertTrue(all(s["status"] == "pending" for s in report["stages"][1:]))

    def test_later_failure_preserves_completed_csvs_and_stops_pipeline(self):
        source = self.root / "source.jsonl"
        video = self.root / "clip.avi"
        video.write_bytes(b"fixture")
        source.write_text(json.dumps({"video": str(video), "text": "a", "windows": [[0, 1]]}) + "\n")
        for failing_stage in ("train_2", "evaluate_2"):
            with self.subTest(stage=failing_stage):
                run_dir = self.root / failing_stage
                expected = []

                def stage(command, log):
                    if log.stem == failing_stage:
                        raise subprocess.CalledProcessError(4, command)
                    if log.stem.startswith("evaluate_"):
                        metrics_path = Path(command[command.index("--output") + 1])
                        train.write_metrics_csv(metrics_path, [{
                            "variant": "first", "sample_index": 0, "text": "a",
                            "count": 1, "avg_nll": 0.25, "avg_margin": 0.5,
                        }])
                        with metrics_path.open(newline="") as stream:
                            expected.extend(csv.DictReader(stream))

                with contextlib.redirect_stdout(io.StringIO()), patch.object(runner, "run_stage", side_effect=stage):
                    with self.assertRaises(subprocess.CalledProcessError):
                        runner.main(["--manifest", str(source), "--run-dir", str(run_dir)])
                report = json.loads((run_dir / "run.json").read_text())
                self.assertEqual(report["status"], "failed")
                failed_index = next(i for i, s in enumerate(report["stages"]) if s["name"] == failing_stage)
                self.assertTrue(all(s["status"] == "complete" for s in report["stages"][:failed_index]))
                self.assertEqual(report["stages"][failed_index]["status"], "failed")
                self.assertTrue(all(s["status"] == "pending" for s in report["stages"][failed_index + 1:]))
                self.assertEqual(len(expected), 1)
                for path in (report["csv"], report["stages"][1]["csv"]):
                    with Path(path).open(newline="") as stream:
                        self.assertEqual(list(csv.DictReader(stream)), expected)

    def test_interrupted_combination_preserves_previous_csv(self):
        previous = self.root / "all_variants.csv"
        previous.write_text("previous results\n")
        metrics_path = self.root / "variant.metrics.csv"
        train.write_metrics_csv(metrics_path, [])

        def interrupted_write(path, rows):
            Path(path).write_text("partial results\n")
            raise KeyboardInterrupt

        with patch.object(train, "write_metrics_csv", side_effect=interrupted_write):
            with self.assertRaises(KeyboardInterrupt):
                runner.combine_metrics_csv([metrics_path], previous)
        self.assertEqual(previous.read_text(), "previous results\n")
        self.assertFalse(previous.with_suffix(".csv.tmp").exists())

    def test_offline_download_handoff_trains_five_and_evaluates_exact_samples(self):
        self.check_offline_download_handoff()

    def test_offline_selected_variants_generate_only_their_metrics(self):
        self.check_offline_download_handoff([5, 3, 5])

    def test_offline_validation_download_training_and_separate_metrics(self):
        self.check_offline_download_handoff([1, 5], validation=True)

    def test_offline_pretraining_is_shared_and_runs_before_supervised_training(self):
        self.check_offline_download_handoff([1, 5], validation=True, pretraining=True)

    def check_offline_download_handoff(self, variants=None, validation=False, pretraining=False):
        selected = list(range(1, 6)) if variants is None else sorted(set(variants))
        video = self.root / "clip.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (32, 32))
        self.assertTrue(writer.isOpened())
        try:
            for frame in range(6):
                writer.write(np.full((32, 32, 3), frame * 30, dtype=np.uint8))
        finally:
            writer.release()
        # Same video path deliberately appears twice with distinct true labels.
        rows = [{"video": str(video), "text": text, "windows": [[0, 2], [3, 5]]} for text in ("ab", "ba")]
        validation_video = self.root / "validation.avi"
        validation_video.write_bytes(video.read_bytes())
        validation_rows = [dict(rows[0], video=str(validation_video), text="ac")]
        actual_run_stage = runner.run_stage
        downloads = []
        completed_metrics = []

        def stage(command, log):
            if Path(command[1]).name == "train_downvid.py":
                downloads.append(command)
                args = train_downvid.build_argparser().parse_args(command[2:])
                if log.stem == "download_pretrain":
                    records = [{"video": str(video)}]
                    destination = args.pretrain_manifest
                else:
                    records = validation_rows if log.stem == "download_validation" else rows
                    destination = args.manifest
                Path(destination).write_text("".join(json.dumps(row) + "\n" for row in records))
                log.write_text("Offline fixture replaces network download/alignment.\n")
            else:
                # Completed CSVs must already be published when the next training starts.
                if log.stem.startswith("train_") and completed_metrics:
                    with (run_dir / "all_variants.csv").open(newline="") as stream:
                        self.assertEqual(list(csv.DictReader(stream)), completed_metrics)
                actual_run_stage(command, log)
                if log.stem.startswith("evaluate_") and not log.stem.startswith("evaluate_validation_"):
                    metrics_path = Path(command[command.index("--output") + 1])
                    with metrics_path.open(newline="") as stream:
                        completed_metrics.extend(csv.DictReader(stream))

        run_dir = self.root / "complete"
        flags = [
            "-N", "2", "--run-dir", str(run_dir), "--device", "cpu", "--epochs", "1",
            "--confidence-epochs", "1", "--batch-size", "2", "--max-frames", "4",
            "--no-face-detector", "--mouth-size", "16", "--conv3d-channels", "8",
            "--d-video", "8", "--d-text", "8", "--d-fusion", "8", "--heads", "2",
            "--video-layers", "1", "--text-layers", "1", "--ff-mult", "1", "--dropout", "0",
            "--log-every", "0",
        ]
        if 4 in selected:
            flags += ["--variant-args", "4:--w-t 0.5 --lr 0.001"]
        if variants is not None:
            flags += ["--variants", *map(str, variants)]
        if validation:
            flags += ["--N-valid", "1"]
        if pretraining:
            flags += ["--N-pretrain", "1", "--pretrain-epochs", "1"]
        output = io.StringIO()
        with patch.dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"), \
                patch.object(runner, "run_stage", side_effect=stage), contextlib.redirect_stdout(output):
            report = runner.main(flags)
        self.assertEqual(len(downloads), 1 + int(validation) + int(pretraining))
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["variants"], selected)
        self.assertEqual(json.loads((run_dir / "run.json").read_text())["variants"], selected)
        expected_stages = ["download", "download_validation"] if validation else ["download"]
        if pretraining:
            expected_stages.append("download_pretrain")
        for index in selected:
            expected_stages += [f"train_{index}", f"evaluate_{index}"]
            if validation:
                expected_stages.append(f"evaluate_validation_{index}")
        self.assertEqual([s["name"] for s in report["stages"]], expected_stages)
        self.assertEqual(len(report["checkpoints"]), len(selected))
        self.assertIn(f"({len(selected)}/{len(selected)} variants)", output.getvalue())
        self.assertIn(f"Completed {len(selected)} selected variant(s) on 2 samples.", output.getvalue())
        self.assertTrue(all(s["status"] == "complete" for s in report["stages"]))
        self.assertEqual(train.load_manifest(report["samples"]["manifest"]), rows)
        with (run_dir / "all_variants.csv").open(newline="") as stream:
            metrics = list(csv.DictReader(stream))
        self.assertEqual(len(metrics), 2 * len(selected))
        self.assertEqual(metrics, completed_metrics)
        self.assertEqual({m["variant"] for m in metrics}, {
            runner.VARIANTS[index - 1].DEFAULTS["variant_name"] for index in selected
        })
        self.assertEqual({str(path) for path in (run_dir / "checkpoints").glob("*.pt")},
                         set(report["checkpoints"]))
        for index, checkpoint in zip(selected, report["checkpoints"]):
            training_stage = next(s for s in report["stages"] if s["name"] == f"train_{index}")
            self.assertTrue(Path(training_stage["learning_plot"]).is_file())
            with Path(training_stage["learning_csv"]).open(newline="") as stream:
                learning = list(csv.DictReader(stream))
            self.assertEqual([row["stage"] for row in learning], ["token", "confidence"])
            for row in learning:
                self.assertTrue(np.isfinite(float(row["training_loss"])))
                self.assertTrue(0 <= float(row["training_accuracy"]) <= 1)
                if validation:
                    self.assertTrue(np.isfinite(float(row["validation_loss"])))
                    self.assertTrue(0 <= float(row["validation_accuracy"]) <= 1)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(saved["training_args"]["manifest"], report["samples"]["manifest"])
            self.assertEqual(saved["training_stage"], "confidence")
            self.assertTrue(saved["confidence"]["trained"])
            if pretraining:
                self.assertEqual(saved["pretrain_epoch"], 1)
                self.assertEqual(saved["training_args"]["pretrain_manifest"], report["pretrain_samples"]["manifest"])
                self.assertEqual(report["pretrain_samples"]["count"], 1)
            if validation:
                self.assertEqual(saved["training_args"]["validation_manifest"], report["validation_samples"]["manifest"])
                self.assertEqual(set(saved["validation_selection"]), {"token", "confidence"})
            selected = [m for m in metrics if m["checkpoint"] == checkpoint]
            self.assertEqual([m["text"] for m in selected], ["ab", "ba"])
            self.assertEqual([m["sample_index"] for m in selected], ["0", "1"])
            self.assertTrue(all(int(m["count"]) == (6 if index == 5 else 2) for m in selected))
            self.assertTrue(all(math_is_valid(m) for m in selected))
            with Path(checkpoint).with_suffix(".metrics.csv").open(newline="") as stream:
                self.assertEqual(list(csv.DictReader(stream)), selected)
        if validation:
            self.assertEqual(train.load_manifest(report["validation_samples"]["manifest"]), validation_rows)
            with Path(report["validation_csv"]).open(newline="") as stream:
                validation_metrics = list(csv.DictReader(stream))
            self.assertEqual(len(validation_metrics), len(report["variants"]))
            self.assertTrue(all(row["text"] == "ac" for row in validation_metrics))
            self.assertTrue(all(math_is_valid(row) for row in validation_metrics))
            self.assertEqual({row["video"] for row in validation_metrics}, {str(validation_video)})


def math_is_valid(row):
    return np.isfinite(float(row["avg_nll"])) and -1 <= float(row["avg_margin"]) <= 1


if __name__ == "__main__":
    unittest.main()
