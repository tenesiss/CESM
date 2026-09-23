"""Streaming accuracy semantics, decoder controls, and learning-curve routing."""

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import infer
import run_all_variants as runner
import train
import train_downvid
from learning_curves import LearningCurveLogger, accuracy_decode_args, token_edit_distance
from streaming import decode_stream


def training_args(*flags):
    return train.build_argparser().parse_args(["--manifest", "unused", "--device", "cpu", *flags])


class ScriptedModel(torch.nn.Module):
    def __init__(self, tokenizer, predictions, confidences=None, *, fusion="residual"):
        super().__init__()
        self.cfg = SimpleNamespace(text_fusion=fusion)
        self.layer = torch.nn.Linear(1, 1)
        self.tokenizer = tokenizer
        self.predictions = iter(predictions)
        self.confidences = iter(confidences) if confidences is not None else None
        self.pushed = []
        self.frames = 0

    def init_stream_state(self):
        return object()

    def stream_push_text_token(self, token, state):
        assert not torch.is_grad_enabled()
        self.pushed.append(int(token.item()))
        return state

    def stream_push_video_frame(self, frame, state):
        self.frames += 1
        value = next(self.confidences) if self.confidences is not None else .99
        return torch.logit(torch.tensor([[value]])), state

    def stream_predict_next(self, state):
        token = next(self.predictions)
        token_id = self.tokenizer.eos_id if token == "<eos>" else self.tokenizer.encode(token)[0]
        scores = torch.full((1, 1, self.tokenizer.vocab_size), -10.0)
        scores[..., token_id] = 10.0
        return scores, None


class StreamingAccuracyTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = train.CharTokenizer.build(["abc"])

    def decode(self, predictions, *flags, confidences=None, frame_count=None, fusion="residual"):
        model = ScriptedModel(self.tokenizer, predictions, confidences, fusion=fusion)
        options = accuracy_decode_args(training_args(
            "--accuracy-warmup-frames", "1", "--accuracy-min-frames-per-token", "0", *flags))
        events = []
        result, count = decode_stream(
            model, self.tokenizer,
            [torch.zeros(3, 16, 16)] * (len(predictions) if frame_count is None else frame_count),
            torch.device("cpu"), options, token_temperature=options.token_temperature,
            conf_temperature=options.conf_temperature, on_event=events.append,
        )
        self.assertEqual(model.pushed, [self.tokenizer.bos_id, *result])
        return self.tokenizer.decode(result), count, events

    def test_warmup_stability_and_frame_gap_apply_together(self):
        text, count, events = self.decode(
            "aaaabbbb", "--accuracy-warmup-frames", "3", "--accuracy-stable-frames", "2",
            "--accuracy-min-frames-per-token", "4")
        self.assertEqual(text, "ab")
        self.assertEqual([e["frame"] for e in events], [3, 7])
        self.assertEqual(count, 8)

    def test_relaxation_accepts_candidate_peak_then_resets_on_change(self):
        flags = ("--accuracy-confidence-threshold", ".9", "--accuracy-confidence-min-threshold", ".5",
                 "--accuracy-confidence-relax-per-frame", ".1", "--accuracy-confidence-relax-after", "0")
        text, _, events = self.decode("aaa", *flags, confidences=[.75, .1, .1])
        self.assertEqual(text, "a")
        self.assertEqual((events[0]["frame"], events[0]["confidence_frame"]), (1, 0))
        text, _, events = self.decode("abb", *flags, confidences=[.75, .1, .1])
        self.assertEqual((text, events), ("", []))

    def test_probability_and_temperature_controls(self):
        text, _, _ = self.decode("a", "--accuracy-min-token-prob", ".99",
                                 "--accuracy-token-temperature", "100")
        self.assertEqual(text, "")
        text, _, _ = self.decode("a", "--accuracy-conf-temperature", "100")
        self.assertEqual(text, "")

    def test_eos_max_tokens_and_frame_flush_exclusion(self):
        self.assertEqual(self.decode(["a", "<eos>", "b"])[:2], ("a", 2))
        self.assertEqual(self.decode("abc", "--accuracy-max-tokens", "1")[:2], ("a", 1))
        self.assertEqual(self.decode("ab", "--accuracy-flush-tokens", "1", frame_count=1)[:2], ("ab", 1))
        self.assertEqual(self.decode("a", "--accuracy-flush-tokens", "1", fusion="frame")[:2], ("a", 1))

    def test_edit_distance_counts_insertions_deletions_substitutions(self):
        for reference, prediction, expected in (("abc", "abc", 0), ("abc", "ac", 1),
                                                 ("abc", "abbc", 1), ("abc", "acc", 1),
                                                 ("", "abc", 3), ("abc", "", 3)):
            self.assertEqual(token_edit_distance(reference, prediction), expected)

    def evaluate(self, references, predictions, *flags):
        dataset = SimpleNamespace(rows=[{"video": f"{i}.avi", "text": text}
                                        for i, text in enumerate(references)],
                                  tokenizer=self.tokenizer, mouth_size=16, use_face_detector=False)
        model = ScriptedModel(self.tokenizer, predictions).train()
        model.layer.eval()
        weights = {key: value.clone() for key, value in model.state_dict().items()}
        captures = []
        for _ in references:
            capture = Mock()
            capture.isOpened.return_value = True
            capture.read.side_effect = [(True, object())] * 3 + [(False, None)]
            captures.append(capture)
        with patch.object(train.cv2, "VideoCapture", side_effect=captures), \
                patch.object(train, "FixedMouthCropper", return_value=lambda frame: torch.zeros(3, 16, 16)):
            score = train.evaluate_streaming_accuracy(model, dataset, torch.device("cpu"), training_args(
                "--accuracy-warmup-frames", "1", "--accuracy-min-frames-per-token", "0", *flags))
        self.assertTrue(model.training)
        self.assertFalse(model.layer.training)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, weights[key], atol=0, rtol=0)
        for capture in captures:
            capture.release.assert_called_once()
        return score, model

    def test_accuracy_aggregates_reference_tokens_and_keeps_free_running_context(self):
        # First video: abc -> a (2 edits); second video: a -> a (0 edits).
        # Corpus accuracy is 1 - 2/4, not mean(1/3, 1).
        score, model = self.evaluate(["abc", "a"], list("aaaaaa"))
        self.assertEqual(score, .5)
        self.assertEqual(model.frames, 6)
        self.assertEqual(model.pushed, [self.tokenizer.bos_id, self.tokenizer.encode("a")[0]] * 2)

    def test_empty_references_insertions_and_floor(self):
        self.assertEqual(self.evaluate([""], ["<eos>"])[0], 1)
        self.assertEqual(self.evaluate([""], list("abc"))[0], 0)
        self.assertEqual(self.evaluate(["a"], list("abc"))[0], 0)


class LearningCurveOptionsTests(unittest.TestCase):
    def test_defaults_match_infer_and_temperatures_follow_training(self):
        actual = accuracy_decode_args(training_args())
        inference = infer.build_argparser().parse_args(["--checkpoint", "unused", "--video", "unused"])
        for key, value in vars(actual).items():
            if key not in ("token_temperature", "conf_temperature"):
                self.assertEqual(value, getattr(inference, key), key)
        options = accuracy_decode_args(training_args("--confidence-token-temperature", "2",
                                                     "--confidence-temperature", "3"))
        self.assertEqual((options.token_temperature, options.conf_temperature), (2, 3))

    def test_runner_routes_positive_negative_and_per_variant_overrides(self):
        parser = runner.build_argparser()
        args = parser.parse_args([
            "--manifest", "unused", "--variants", "1", "5", "--no-plot-training-accuracy",
            "--accuracy-warmup-frames", "8", "--accuracy-repeat-token-cooldown-frames", "0",
            "--variant-args", "5:--plot-training-accuracy --no-plot-validation-loss --accuracy-stable-frames 3",
        ])
        _, _, _, stages = runner.build_commands(parser, args, Path("run"))
        first = train.build_argparser().parse_args(stages[0]["command"][2:])
        last = train.build_argparser().parse_args(stages[2]["command"][2:])
        self.assertFalse(first.plot_training_accuracy)
        self.assertTrue(last.plot_training_accuracy)
        self.assertFalse(last.plot_validation_loss)
        self.assertEqual((first.accuracy_warmup_frames, last.accuracy_warmup_frames), (8, 8))
        self.assertEqual(last.accuracy_repeat_token_cooldown_frames, 0)
        self.assertEqual(last.accuracy_stable_frames, 3)
        self.assertTrue(stages[0]["learning_plot"].endswith(".learning.png"))
        args = parser.parse_args(["-N", "2", "--no-plot-learning-curves"])
        _, _, _, stages = runner.build_commands(parser, args, Path("run"))
        self.assertTrue(all("learning_plot" not in stage for stage in stages))

    def test_downloader_boolean_flags_round_trip(self):
        args = train_downvid.build_argparser().parse_args([
            "-N", "1", "--manifest", "unused", "--no-plot-training-loss", "--no-plot-learning-curves"])
        actual = train.build_argparser().parse_args(train_downvid.training_command(args)[2:])
        self.assertFalse(actual.plot_training_loss)
        self.assertFalse(actual.plot_learning_curves)

    def test_invalid_decoder_values_fail_before_model_creation(self):
        cases = (("warmup-frames", "-1"), ("repeat-token-cooldown-frames", "-1"),
                 ("stable-frames", "0"), ("max-tokens", "0"), ("flush-tokens", "-1"),
                 ("confidence-threshold", "nan"), ("confidence-min-threshold", ".9"),
                 ("confidence-relax-per-frame", "inf"), ("conf-temperature", "0"))
        for option, value in cases:
            with self.subTest(option=option), patch.object(train, "UnnobaModel") as model:
                with self.assertRaisesRegex(ValueError, "--accuracy-"):
                    train.train(training_args("--accuracy-" + option, value))
                model.assert_not_called()

    def test_csv_and_plot_only_selected_curves_with_separate_stage_panels(self):
        with tempfile.TemporaryDirectory() as temp:
            args = training_args("--output", str(Path(temp) / "model.pt"),
                                 "--validation-manifest", "validation", "--no-plot-training-accuracy")
            logger = LearningCurveLogger(args)
            for stage, loss in (("token", 2.0), ("confidence", .1)):
                logger.record(stage, 1, stage == "confidence", training_loss=loss,
                              validation_loss=loss + .1, training_accuracy=.2, validation_accuracy=.3)
            with logger.csv_path.open(newline="") as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual([row["stage"] for row in records], ["token", "confidence"])
            self.assertEqual([row["training_accuracy"] for row in records], ["", ""])
            self.assertEqual([row["validation_accuracy"] for row in records], ["0.3", "0.3"])
            self.assertEqual(logger.plot_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_missing_validation_and_all_disabled_select_no_curves(self):
        args = training_args("--no-plot-training-loss", "--no-plot-training-accuracy")
        self.assertEqual(LearningCurveLogger(args).selected, [])
        args = training_args("--no-plot-learning-curves", "--validation-manifest", "validation")
        self.assertEqual(LearningCurveLogger(args).selected, [])


if __name__ == "__main__":
    unittest.main()
