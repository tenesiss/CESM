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
import train_talkvid


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
        preparation = train_talkvid.build_argparser().parse_args(stages[0]["command"][2:])
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
        prepare = train_talkvid.build_argparser().parse_args(stages[0]["command"][2:])
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

    def test_invalid_or_dataset_changing_overrides_fail_before_download(self):
        for flags in (
            ["-N", "1", "--variant-args", "3:--manifest other.jsonl"],
            ["-N", "1", "--variant-args", "4:--pretrained-lm other"],
            ["-N", "1", "--variant-args", "5:--lambda-tcross 1"],
            ["-N", "1", "--variant-args", "1:--text-fusion frame"],
            ["-N", "1", "--epochs", "0"],
            ["-N", "1", "--lr", "nan"],
            ["-N", "1", "--batch-size", "0"],
            ["-N", "1", "--manifest", "existing.jsonl"],
        ):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.plan(flags)

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
                args = train_talkvid.build_argparser().parse_args(command[2:])
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
        actual_run_stage = runner.run_stage
        downloads = []
        completed_metrics = []

        def stage(command, log):
            if Path(command[1]).name == "train_talkvid.py":
                downloads.append(command)
                args = train_talkvid.build_argparser().parse_args(command[2:])
                Path(args.manifest).write_text("".join(json.dumps(row) + "\n" for row in rows))
                log.write_text("Offline fixture replaces network download/alignment.\n")
            else:
                # Completed CSVs must already be published when the next training starts.
                if log.stem.startswith("train_") and completed_metrics:
                    with (run_dir / "all_variants.csv").open(newline="") as stream:
                        self.assertEqual(list(csv.DictReader(stream)), completed_metrics)
                actual_run_stage(command, log)
                if log.stem.startswith("evaluate_"):
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
            "--log-every", "0", "--variant-args", "4:--w-t 0.5 --lr 0.001",
        ]
        with patch.dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"), \
                patch.object(runner, "run_stage", side_effect=stage), contextlib.redirect_stdout(io.StringIO()):
            report = runner.main(flags)
        self.assertEqual(len(downloads), 1)
        self.assertEqual(report["status"], "complete")
        self.assertTrue(all(s["status"] == "complete" for s in report["stages"]))
        self.assertEqual(train.load_manifest(report["samples"]["manifest"]), rows)
        with (run_dir / "all_variants.csv").open(newline="") as stream:
            metrics = list(csv.DictReader(stream))
        self.assertEqual(len(metrics), 10)
        self.assertEqual(metrics, completed_metrics)
        for index, checkpoint in enumerate(report["checkpoints"], 1):
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(saved["training_args"]["manifest"], report["samples"]["manifest"])
            self.assertEqual(saved["training_stage"], "confidence")
            self.assertTrue(saved["confidence"]["trained"])
            selected = [m for m in metrics if m["checkpoint"] == checkpoint]
            self.assertEqual([m["text"] for m in selected], ["ab", "ba"])
            self.assertEqual([m["sample_index"] for m in selected], ["0", "1"])
            self.assertTrue(all(int(m["count"]) == (6 if index == 5 else 2) for m in selected))
            self.assertTrue(all(math_is_valid(m) for m in selected))
            with Path(checkpoint).with_suffix(".metrics.csv").open(newline="") as stream:
                self.assertEqual(list(csv.DictReader(stream)), selected)


def math_is_valid(row):
    return np.isfinite(float(row["avg_nll"])) and -1 <= float(row["avg_margin"]) <= 1


if __name__ == "__main__":
    unittest.main()
