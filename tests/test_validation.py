"""Held-out loss evaluation, patience, and restoration of selected model weights."""

import contextlib
import csv
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch

import train


class ValidationSplitTests(unittest.TestCase):
    def test_count_is_positive_and_exclusive_with_a_manifest(self):
        for flags in (["--N-valid", "0"], ["--N-valid", "-1"], ["--N-valid", "1.5"],
                      ["--N-valid", "1", "--validation-manifest", "valid.jsonl"]):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train.build_argparser().parse_args(flags)

    def test_seeded_split_is_exact_disjoint_and_keeps_original_order(self):
        rows = [{"video": f"/tmp/clip-{i}.mp4"} for i in range(12)]
        state = random.getstate()
        training, validation = train.split_validation_rows(rows, 4, 42)
        self.assertEqual((len(training), len(validation)), (8, 4))
        self.assertEqual((training, validation), train.split_validation_rows(rows, 4, 42))
        self.assertNotEqual(validation, train.split_validation_rows(rows, 4, 43)[1])
        self.assertEqual(random.getstate(), state)
        self.assertEqual(validation, [row for row in rows if row not in training])
        train.validate_validation_split(training, validation)

    def test_connected_paths_and_sources_are_never_split(self):
        # Groups have sizes 3 and 2; a greedy selection of the pair would fail
        # even though holding out exactly three rows is possible.
        rows = [{"video": "/tmp/a.mp4", "source_key": "one"},
                {"video": "/tmp/b.mp4", "source_key": "one"},
                {"video": "/tmp/b.mp4", "source_key": "two"},
                {"video": "/tmp/c.mp4", "source_key": "three"},
                {"video": "/tmp/d.mp4", "source_key": "three"}]
        for seed in range(4):
            training, validation = train.split_validation_rows(rows, 3, seed)
            self.assertEqual(validation, rows[:3])
            train.validate_validation_split(training, validation)
        with self.assertRaisesRegex(ValueError, "without splitting"):
            train.split_validation_rows(rows, 1, 42)

    def test_pretraining_sources_are_excluded_and_impossible_counts_fail(self):
        rows = [{"video": f"/tmp/clip-{i}.mp4", "source_key": str(i)} for i in range(3)]
        pretraining = [{"video": "/tmp/pretrain.mp4", "source_key": "0"}, rows[1]]
        training, validation = train.split_validation_rows(rows, 1, 42, excluded_rows=pretraining)
        self.assertEqual(validation, [rows[2]])
        train.validate_validation_split(training + pretraining, validation)
        with self.assertRaisesRegex(ValueError, "overlapping pretraining"):
            train.split_validation_rows(rows, 2, 42, excluded_rows=pretraining)
        for count in (0, 3, 4):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "smaller than"):
                train.split_validation_rows(rows, count, 42)


class ValidationMonitorTests(unittest.TestCase):
    def test_patience_monitors_validation_but_selection_uses_both_losses(self):
        monitor = train.ValidationMonitor(patience=2, min_delta=0.1)
        self.assertTrue(monitor.update(1, 4.0, 2.0))
        self.assertTrue(monitor.update(2, 1.0, 2.1))
        self.assertFalse(monitor.should_stop)
        self.assertFalse(monitor.update(3, 2.0, 2.2))
        self.assertTrue(monitor.should_stop)
        self.assertEqual(monitor.best["epoch"], 2)
        self.assertEqual(monitor.best["score"], 1.55)

    def test_min_delta_only_affects_patience_and_improvement_resets_it(self):
        monitor = train.ValidationMonitor(patience=2, min_delta=0.1)
        monitor.update(1, 2.0, 2.0)
        self.assertTrue(monitor.update(2, 1.0, 1.95))
        self.assertEqual(monitor.bad_epochs, 1)
        monitor.update(3, 1.0, 1.8)
        self.assertEqual(monitor.bad_epochs, 0)
        disabled = train.ValidationMonitor(patience=0, min_delta=0.0)
        for epoch in range(5):
            disabled.update(epoch, 1.0, 1.0)
        self.assertFalse(disabled.should_stop)
        self.assertEqual(disabled.best["epoch"], 0)  # Ties retain the earlier model.

    def test_nonfinite_losses_are_not_selected(self):
        monitor = train.ValidationMonitor(1, 0.0)
        monitor.update(1, 1.0, 1.0)
        for losses in ((float("nan"), 1.0), (1.0, float("inf"))):
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                monitor.update(2, *losses)
        self.assertEqual(len(monitor.history), 1)

    def test_loss_evaluation_weights_positions_and_preserves_modes_and_weights(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = SimpleNamespace(text_fusion="residual")
                self.layer = torch.nn.Linear(1, 1)

            def forward(self, batch, **unused):
                return {"token_logits": self.layer(batch["features"])}

        model = Model().train()
        model.layer.eval()  # Mixed modes, as in the confidence training stage.
        weights = {name: value.clone() for name, value in model.state_dict().items()}
        batches = [
            {"features": torch.ones(1, 1), "targets": torch.tensor([[1, 0, 0]])},
            {"features": torch.ones(1, 1), "targets": torch.tensor([[1, 2, 3]])},
            {"features": torch.ones(1, 1), "targets": torch.tensor([[0, 0, 0]])},
        ]
        args = SimpleNamespace(pad_id=0, amp=False, flash_mono=False, lambda_mono=0, lambda_align=0)
        values = iter([1.0, 3.0])

        def objective(model, batch, out, args):
            self.assertFalse(model.training)
            self.assertFalse(torch.is_grad_enabled())
            return {"total": torch.tensor(next(values))}

        with patch.object(train, "compute_losses", side_effect=objective) as compute:
            loss = train.evaluate_loss(model, batches, torch.device("cpu"), args)
        self.assertEqual(loss, 2.5)  # (1 position * 1 + 3 positions * 3) / 4
        self.assertEqual(compute.call_count, 2)
        self.assertTrue(model.training)
        self.assertFalse(model.layer.training)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, weights[name], atol=0, rtol=0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_overlap_checks_both_paths_and_source_uploads(self):
        training = [{"video": "/tmp/training.mp4", "source_key": "upload-1"}]
        for validation in ([{"video": "/tmp/training.mp4"}],
                           [{"video": "/tmp/another_clip.mp4", "source_key": "upload-1"}]):
            with self.assertRaisesRegex(ValueError, "overlap"):
                train.validate_validation_split(training, validation)
        train.validate_validation_split(training, [{"video": "/tmp/validation.mp4", "source_key": "upload-2"}])


class ValidationTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for split, text in (("train", "ab"), ("valid", "ac")):
            video = self.root / f"{split}.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (32, 32))
            self.assertTrue(writer.isOpened())
            try:
                for frame in range(6):
                    writer.write(np.full((32, 32, 3), frame * 20, dtype=np.uint8))
            finally:
                writer.release()
            (self.root / f"{split}.jsonl").write_text(json.dumps({
                "video": str(video), "text": text, "windows": [[0, 2], [3, 5]],
            }) + "\n")

    def args(self, *extra):
        return train.build_argparser().parse_args([
            "--manifest", str(self.root / "train.jsonl"),
            "--validation-manifest", str(self.root / "valid.jsonl"),
            "--output", str(self.root / "model.pt"), "--device", "cpu",
            "--epochs", "5", "--confidence-epochs", "4", "--batch-size", "1",
            "--early-stopping-patience", "1", "--no-face-detector", "--mouth-size", "16",
            "--conv3d-channels", "8", "--d-video", "8", "--d-text", "8", "--d-fusion", "8",
            "--heads", "2", "--video-layers", "1", "--text-layers", "1", "--ff-mult", "1",
            "--dropout", "0", "--log-every", "0", "--metrics-csv", "auto", *extra,
        ])

    def assert_weights_equal(self, actual, expected):
        self.assertEqual(actual.keys(), expected.keys())
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0, msg=name)

    def test_n_valid_trains_only_remaining_rows_and_records_both_validation_stages(self):
        manifest = self.root / "combined.jsonl"
        manifest.write_text((self.root / "train.jsonl").read_text() + (self.root / "valid.jsonl").read_text())
        args = self.args("--manifest", str(manifest), "--epochs", "1", "--confidence-epochs", "1")
        args.validation_manifest = None
        args.num_valid = 1
        training, validation = train.split_validation_rows(train.load_manifest(manifest), 1, args.seed)
        with contextlib.redirect_stdout(io.StringIO()):
            train.train(args)
        saved = torch.load(args.output, map_location="cpu", weights_only=False)
        self.assertEqual(saved["training_args"]["num_valid"], 1)
        self.assertEqual(set(saved["validation_selection"]), {"token", "confidence"})
        with Path(args.output).with_suffix(".metrics.csv").open() as stream:
            metrics = list(csv.DictReader(stream))
        self.assertEqual([row["video"] for row in metrics], [row["video"] for row in training])
        with Path(args.output).with_suffix(".learning.csv").open() as stream:
            curves = list(csv.DictReader(stream))
        self.assertEqual([row["stage"] for row in curves], ["token", "confidence"])
        self.assertTrue(all(row["validation_loss"] and row["validation_accuracy"] for row in curves))
        self.assertTrue(Path(args.output).with_suffix(".learning.png").is_file())
        self.assertEqual(len(train.load_manifest(manifest)), 2)  # Input remains intact.
        args.resume = args.output
        args.epochs = 0
        with contextlib.redirect_stdout(io.StringIO()):
            train.train(args)
        with Path(args.output).with_suffix(".metrics.csv").open() as stream:
            self.assertEqual([row["video"] for row in csv.DictReader(stream)], [training[0]["video"]])

    def test_invalid_holdout_fails_before_model_creation(self):
        for count, pretrain_only, message in ((1, False, "smaller than"), (1, True, "pretrain-only")):
            args = self.args()
            args.validation_manifest = None
            args.num_valid = count
            args.pretrain_only = pretrain_only
            with self.subTest(pretrain_only=pretrain_only), patch.object(train, "UnnobaModel") as model, \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, message):
                train.train(args)
            model.assert_not_called()

    def test_both_stages_stop_restore_weights_and_metrics_use_best_checkpoint(self):
        for fusion in ("residual", "frame"):
            with self.subTest(fusion=fusion):
                flags = ["--text-fusion", fusion, "--max-frames", "4"]
                if fusion == "frame":
                    flags += ["--window-phasing", "0.5"]
                args = self.args(*flags)
                # Validation improves once, then worsens; the final epoch must be discarded.
                scores = iter([3.0, 3.0, 1.0, 2.0, 4.0, 3.0, 1.0, 1.0, 2.0, 2.0])
                snapshots = {"token": [], "confidence": []}
                actual_losses = []
                evaluate = train.evaluate_loss
                configure = train.configure_confidence_stage
                metrics = train.evaluate_samples

                def measured(model, loader, device, args, *, stage="token"):
                    # Run the actual loss evaluation too, including frame/confidence objectives.
                    actual_losses.append(evaluate(model, loader, device, args, stage=stage))
                    if loader.dataset.rows[0]["text"] == "ab":
                        snapshots[stage].append({k: v.detach().clone() for k, v in model.state_dict().items()})
                    return next(scores)

                def confidence(model):
                    self.assert_weights_equal(model.state_dict(), snapshots["token"][1])
                    return configure(model)

                def evaluate_metrics(model, *positional, **kwargs):
                    self.assert_weights_equal(model.state_dict(), snapshots["confidence"][0])
                    return metrics(model, *positional, **kwargs)

                with patch.object(train, "evaluate_loss", side_effect=measured), \
                        patch.object(train, "configure_confidence_stage", side_effect=confidence), \
                        patch.object(train, "evaluate_samples", side_effect=evaluate_metrics), \
                        contextlib.redirect_stdout(io.StringIO()):
                    train.train(args)
                saved = torch.load(args.output, map_location="cpu", weights_only=False)
                self.assertEqual((saved["epoch"], saved["confidence_epoch"]), (2, 1))
                self.assertTrue(saved["confidence"]["trained"])
                self.assert_weights_equal(saved["model_state"], snapshots["confidence"][0])
                for stage, count in (("token", 3), ("confidence", 2)):
                    history = saved["validation_selection"][stage]
                    self.assertTrue(history["stopped_early"])
                    self.assertEqual(len(history["history"]), count)
                self.assertTrue(all(np.isfinite(actual_losses)))
                self.assertNotIn("c", saved["tokenizer"]["itos"])  # Vocabulary comes only from training.
                with Path(args.output).with_suffix(".metrics.csv").open() as stream:
                    self.assertEqual([row["text"] for row in csv.DictReader(stream)], ["ab"])
                with Path(args.output).with_suffix(".learning.csv").open() as stream:
                    learning = list(csv.DictReader(stream))
                self.assertEqual([row["stage"] for row in learning], ["token"] * 3 + ["confidence"] * 2)
                self.assertEqual([float(row["training_loss"]) for row in learning], [3, 1, 4, 1, 2])
                self.assertEqual([float(row["validation_loss"]) for row in learning], [3, 2, 3, 1, 2])
                for row in learning:
                    for key in ("training_accuracy", "validation_accuracy"):
                        self.assertTrue(0 <= float(row[key]) <= 1)
                self.assertEqual(learning[0]["confidence_trained"], "False")
                self.assertEqual(learning[-1]["confidence_trained"], "True")
                self.assertTrue(Path(args.output).with_suffix(".learning.png").is_file())

    def test_epoch_limit_also_restores_best_without_early_stopping(self):
        args = self.args("--epochs", "2", "--confidence-epochs", "0", "--early-stopping-patience", "0")
        scores = iter([1.0, 1.0, 2.0, 2.0])
        snapshots = []

        def measured(model, *unused, **kwargs):
            snapshots.append({k: v.detach().clone() for k, v in model.state_dict().items()})
            return next(scores)

        with patch.object(train, "evaluate_loss", side_effect=measured), contextlib.redirect_stdout(io.StringIO()):
            train.train(args)
        saved = torch.load(args.output, map_location="cpu", weights_only=False)
        self.assertEqual(saved["epoch"], 1)
        self.assertFalse(saved["validation_selection"]["token"]["stopped_early"])
        self.assert_weights_equal(saved["model_state"], snapshots[0])

    def test_overlap_and_invalid_patience_fail_before_model_creation(self):
        for flags, message in ((["--validation-manifest", str(self.root / "train.jsonl")], "overlap"),
                               (["--early-stopping-patience", "-1"], "patience"),
                               (["--early-stopping-min-delta", "nan"], "min-delta")):
            with self.subTest(flags=flags), patch.object(train, "UnnobaModel") as model:
                with self.assertRaisesRegex(ValueError, message):
                    train.train(self.args(*flags))
                model.assert_not_called()

    def test_disabled_curves_skip_extra_evaluation_and_keep_final_metrics(self):
        args = self.args("--epochs", "1", "--confidence-epochs", "0", "--validation-manifest", "",
                         "--no-plot-learning-curves")
        with patch.object(train, "evaluate_streaming_accuracy") as accuracy, \
                patch.object(train, "evaluate_loss") as loss, contextlib.redirect_stdout(io.StringIO()):
            train.train(args)
        accuracy.assert_not_called()
        loss.assert_not_called()
        self.assertFalse(Path(args.output).with_suffix(".learning.csv").exists())
        self.assertTrue(Path(args.output).with_suffix(".metrics.csv").exists())

    def test_no_validation_creates_training_curves_and_resume_starts_fresh_history(self):
        args = self.args("--epochs", "1", "--confidence-epochs", "0", "--validation-manifest", "")
        with contextlib.redirect_stdout(io.StringIO()):
            train.train(args)
        with Path(args.output).with_suffix(".learning.csv").open() as stream:
            records = list(csv.DictReader(stream))
        self.assertEqual(len(records), 1)
        self.assertTrue(float(records[0]["training_loss"]) > 0)
        self.assertEqual((records[0]["validation_loss"], records[0]["validation_accuracy"]), ("", ""))
        args.resume = args.output
        args.epochs = 0
        args.confidence_epochs = 1
        with contextlib.redirect_stdout(io.StringIO()):
            train.train(args)
        with Path(args.output).with_suffix(".learning.csv").open() as stream:
            records = list(csv.DictReader(stream))
        self.assertEqual(len(records), 1)
        self.assertEqual((records[0]["stage"], records[0]["epoch"]), ("confidence", "1"))

    def test_individual_curve_toggles_skip_training_decoding_but_keep_validation_selection(self):
        args = self.args("--epochs", "1", "--confidence-epochs", "0",
                         "--no-plot-training-accuracy", "--no-plot-validation-loss")
        with patch.object(train, "evaluate_streaming_accuracy", wraps=train.evaluate_streaming_accuracy) as accuracy, \
                patch.object(train, "evaluate_loss", wraps=train.evaluate_loss) as loss, \
                contextlib.redirect_stdout(io.StringIO()):
            train.train(args)
        self.assertEqual(accuracy.call_count, 1)
        self.assertEqual(accuracy.call_args.args[1].rows[0]["text"], "ac")
        self.assertEqual(loss.call_count, 2)  # Both losses are still needed for checkpoint selection.
        with Path(args.output).with_suffix(".learning.csv").open() as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual((row["training_accuracy"], row["validation_loss"]), ("", ""))
        self.assertTrue(0 <= float(row["validation_accuracy"]) <= 1)


if __name__ == "__main__":
    unittest.main()
