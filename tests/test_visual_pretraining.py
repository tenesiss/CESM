"""Masked self-supervision, stochastic views, and pretraining-stage integration."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import train


class VisualPretrainingLossTests(unittest.TestCase):
    def test_temporal_pairs_are_unique_and_weighted_by_valid_pair_count(self):
        z = torch.tensor([[[0., 0.], [1., 3.], [4., 6.], [8., 10.]],
                          [[2., 4.], [5., 8.], [float("nan"), 0.], [float("nan"), 0.]]],
                         requires_grad=True)
        lengths = torch.tensor([4, 2])
        for adjacent in (2, 3, 20):
            expected = torch.stack([
                (z[b, i] - z[b, j]).square().mean()
                for b, length in enumerate(lengths.tolist())
                for i in range(length) for j in range(i + 1, length)
                if j - i < adjacent
            ]).mean()
            actual = train.temporal_visual_loss(z, lengths, adjacent)
            torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertEqual(z.grad[1, 2:].abs().sum().item(), 0)

    def test_single_frame_has_differentiable_zero_temporal_loss(self):
        z = torch.randn(2, 1, 4, requires_grad=True)
        loss = train.temporal_visual_loss(z, torch.ones(2, dtype=torch.long))
        self.assertEqual(loss.item(), 0)
        loss.backward()
        torch.testing.assert_close(z.grad, torch.zeros_like(z))

    def test_augmentation_loss_masks_padding_and_trains_both_views(self):
        z1 = torch.tensor([[[1., 3.], [4., 8.]], [[6., 8.], [float("nan"), 0.]]],
                          requires_grad=True)
        z2 = torch.zeros_like(z1, requires_grad=True)
        lengths = torch.tensor([2, 1])
        loss = train.augmentation_visual_loss(z1, z2, lengths)
        self.assertAlmostEqual(loss.item(), (5 + 40 + 50) / 3, places=5)
        loss.backward()
        for z in (z1, z2):
            self.assertTrue(torch.isfinite(z.grad).all())
            self.assertGreater(z.grad[0].abs().sum().item(), 0)
            self.assertEqual(z.grad[1, 1].abs().sum().item(), 0)

    def test_mse_uses_fp32_for_half_precision_representations(self):
        z = torch.tensor([[[300.], [-300.]]], dtype=torch.float16, requires_grad=True)
        loss = train.temporal_visual_loss(z, torch.tensor([2]))
        self.assertEqual(loss.dtype, torch.float32)
        self.assertEqual(loss.item(), 360000)

    def test_variance_and_covariance_distinguish_collapse_redundancy_and_diversity(self):
        lengths = torch.tensor([4])
        collapsed = torch.tensor([[[2., -3.]] * 4])
        redundant = torch.tensor([[[-1., -1.], [-1., -1.], [1., 1.], [1., 1.]]])
        diverse = torch.tensor([[[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]]])
        variance, covariance = train.visual_variance_covariance_losses(collapsed, lengths)
        self.assertAlmostEqual(variance.item(), 0.99, places=6)
        self.assertEqual(covariance.item(), 0)
        variance, covariance = train.visual_variance_covariance_losses(redundant, lengths)
        self.assertEqual(variance.item(), 0)
        self.assertAlmostEqual(covariance.item(), 16 / 9, places=6)
        variance, covariance = train.visual_variance_covariance_losses(diverse, lengths)
        self.assertEqual(variance.item(), 0)
        self.assertEqual(covariance.item(), 0)
        # A small user-selected floor must not be swallowed by numerical epsilon.
        variance, _ = train.visual_variance_covariance_losses(collapsed, lengths, 0.001)
        self.assertGreater(variance.item(), 0)

    def test_statistics_pool_valid_frames_mask_padding_and_match_sample_covariance(self):
        z = torch.tensor([[[-1., -1.], [0., 0.]], [[1., 1.], [float("nan"), float("nan")]]],
                         requires_grad=True)
        lengths = torch.tensor([2, 1])
        variance, covariance = train.visual_variance_covariance_losses(z, lengths, 2)
        self.assertAlmostEqual(variance.item(), 2 - (1 + 1e-4) ** 0.5, places=6)
        self.assertAlmostEqual(covariance.item(), 1, places=6)
        shifted = train.visual_variance_covariance_losses(z + 10, lengths, 2)
        torch.testing.assert_close((variance, covariance), shifted)
        (variance + covariance).backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(z.grad[0].abs().sum().item(), 0)
        self.assertEqual(z.grad[1, 1].abs().sum().item(), 0)

    def test_variance_gradient_expands_nearly_collapsed_features(self):
        z = torch.tensor([[[-0.1], [0.1]]], requires_grad=True)
        variance, covariance = train.visual_variance_covariance_losses(z, torch.tensor([2]))
        variance.backward()
        self.assertGreater(z.grad[0, 0, 0].item(), 0)
        self.assertLess(z.grad[0, 1, 0].item(), 0)
        self.assertEqual(covariance.item(), 0)  # A single feature has no off-diagonal entries.

    def test_statistics_skip_batches_with_fewer_than_two_valid_frames(self):
        for length in (0, 1):
            with self.subTest(length=length):
                z = torch.tensor([[[3., 4.], [float("nan"), float("nan")]]], requires_grad=True)
                variance, covariance = train.visual_variance_covariance_losses(z, torch.tensor([length]))
                self.assertEqual(variance.item(), 0)
                self.assertEqual(covariance.item(), 0)
                (variance + covariance).backward()
                torch.testing.assert_close(z.grad, torch.zeros_like(z))

    def test_statistics_remain_fp32_under_autocast(self):
        z = torch.tensor([[[-300., -300.], [300., 300.]]], dtype=torch.float16, requires_grad=True)
        lengths = torch.tensor([2])
        expected = train.visual_variance_covariance_losses(z.float(), lengths, 500)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = train.visual_variance_covariance_losses(z, lengths, 500)
        for result in actual:
            self.assertEqual(result.dtype, torch.float32)
            self.assertTrue(torch.isfinite(result))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertAlmostEqual(actual[1].item() / 180000 ** 2, 1, places=6)

    def test_weighted_objective_aggregates_views_and_backpropagates_to_each_pass(self):
        clean = torch.tensor([[[0., 0.], [2., 2.]]], requires_grad=True)
        z1 = torch.tensor([[[-1., -1.], [1., 1.]]], requires_grad=True)
        z2 = torch.tensor([[[-2., -2.], [2., 2.]]], requires_grad=True)
        encoder = Mock(side_effect=[clean, z1, z2])
        batch = {"video": torch.zeros(1, 2, 3, 8, 8), "video_lengths": torch.tensor([2])}
        losses = train.visual_pretraining_losses(
            encoder, batch, lambda_temporal=2, lambda_augmentation=3,
            lambda_variance=5, lambda_covariance=7, variance_floor=4,
        )
        expected_variance = 4 - ((2 + 1e-4) ** 0.5 + (8 + 1e-4) ** 0.5) / 2
        self.assertEqual(losses["temporal"].item(), 4)
        self.assertEqual(losses["augmentation"].item(), 1)
        self.assertAlmostEqual(losses["variance"].item(), expected_variance, places=6)
        self.assertEqual(losses["covariance"].item(), 4 + 64)
        self.assertAlmostEqual(losses["total"].item(), 2 * 4 + 3 + 5 * expected_variance + 7 * 68, places=4)
        losses["total"].backward()
        for z in (clean, z1, z2):
            self.assertTrue(torch.isfinite(z.grad).all())
            self.assertGreater(z.grad.abs().sum().item(), 0)

    def test_disabled_temporal_skips_clean_pass_and_legacy_weights_reproduce_mse_sum(self):
        z1 = torch.tensor([[[0., 0.], [2., 2.]]], requires_grad=True)
        z2 = torch.zeros_like(z1, requires_grad=True)
        batch = {"video": torch.zeros(1, 2, 3, 8, 8), "video_lengths": torch.tensor([2])}
        encoder = Mock(side_effect=[z1, z2])
        losses = train.visual_pretraining_losses(encoder, batch)
        self.assertEqual(encoder.call_count, 2)
        self.assertEqual(losses["temporal"].item(), 0)
        encoder = Mock(side_effect=[z1, z1, z2])
        with patch.object(train, "visual_variance_covariance_losses") as statistics:
            losses = train.visual_pretraining_losses(
                encoder, batch, lambda_temporal=1, lambda_augmentation=1,
                lambda_variance=0, lambda_covariance=0,
            )
        statistics.assert_not_called()
        self.assertEqual(losses["total"].item(), 4 + 2)
        self.assertEqual(losses["variance"].item(), 0)
        self.assertEqual(losses["covariance"].item(), 0)

    def test_original_probability_controls_both_pretraining_views(self):
        torch.manual_seed(17)
        video = torch.rand(2, 3, 3, 8, 8) * 2 - 1
        video[1, 1:] = float("nan")
        lengths = torch.tensor([3, 1])
        valid = torch.arange(3)[None, :] < lengths[:, None]
        for probability in (0, 1):
            with self.subTest(probability=probability):
                encoder = Mock(side_effect=lambda view, lengths: view.mean(dim=(2, 3, 4)).unsqueeze(-1))
                train.visual_pretraining_losses(
                    encoder, {"video": video, "video_lengths": lengths}, original_probability=probability,
                )
                self.assertEqual(encoder.call_count, 2)
                for call in encoder.call_args_list:
                    view = call.args[0]
                    if probability == 1:
                        torch.testing.assert_close(view[valid], video[valid], rtol=0, atol=0)
                    else:
                        self.assertFalse((view[valid] == video[valid]).flatten(1).all(1).any())
                    self.assertEqual(view[~valid].abs().sum().item(), 0)

    def test_views_change_each_call_retain_originals_and_preserve_padding(self):
        torch.manual_seed(12)
        video = torch.rand(2, 50, 3, 8, 8) * 2 - 1
        lengths = torch.tensor([50, 23])
        video[1, 23:] = float("nan")
        before = video.clone()
        view1 = train.augment_pretrain_video(video, lengths)
        view2 = train.augment_pretrain_video(video, lengths)
        self.assertFalse(torch.equal(view1, view2))
        valid = torch.arange(50)[None, :] < lengths[:, None]
        originals = (view1[valid] == video[valid]).flatten(1).all(1)
        self.assertTrue(originals.any())
        self.assertTrue((~originals).any())
        self.assertTrue(torch.isfinite(view1).all())
        self.assertLessEqual(view1.max().item(), 1)
        self.assertGreaterEqual(view1.min().item(), -1)
        self.assertEqual(view1[1, 23:].abs().sum().item(), 0)
        torch.testing.assert_close(video, before, equal_nan=True)
        identity = train.augment_pretrain_video(video.half(), lengths, original_probability=1)
        self.assertEqual(identity.dtype, torch.float16)
        torch.testing.assert_close(identity[valid], video[valid].half(), rtol=0, atol=0)
        torch.manual_seed(99)
        seeded1 = train.augment_pretrain_video(video, lengths)
        torch.manual_seed(99)
        torch.testing.assert_close(seeded1, train.augment_pretrain_video(video, lengths))


class VisualPretrainingTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        writer = cv2.VideoWriter(str(self.root / "clip.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 25, (32, 32))
        self.assertTrue(writer.isOpened())
        try:
            rng = np.random.default_rng(7)
            for _ in range(5):
                writer.write(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
        finally:
            writer.release()
        (self.root / "train.jsonl").write_text(json.dumps({
            "video": "clip.avi", "text": "ab", "windows": [[0, 1], [2, 4]],
        }) + "\n")
        (self.root / "pretrain.jsonl").write_text(json.dumps({"video": "clip.avi"}) + "\n")

    def args(self, *extra):
        return train.build_argparser().parse_args([
            "--manifest", str(self.root / "train.jsonl"),
            "--pretrain-visual-encoder", "--pretrain-manifest", str(self.root / "pretrain.jsonl"),
            "--pretrain-epochs", "1", "--output", str(self.root / "model.pt"),
            "--device", "cpu", "--epochs", "0", "--confidence-epochs", "0",
            "--batch-size", "2", "--max-frames", "3", "--no-face-detector", "--mouth-size", "16",
            "--conv3d-channels", "8", "--d-video", "8", "--d-text", "8", "--d-fusion", "8",
            "--heads", "2", "--video-layers", "1", "--text-layers", "1", "--ff-mult", "1",
            "--dropout", "0", "--log-every", "1", "--no-plot-learning-curves", *extra,
        ])

    def test_video_only_manifest_sectioning_and_collation(self):
        rows = train.load_manifest(str(self.root / "pretrain.jsonl"), video_only=True)
        self.assertEqual(rows, [{"video": str(self.root / "clip.avi")}])
        with self.assertRaisesRegex(ValueError, "missing key 'text'"):
            train.load_manifest(str(self.root / "pretrain.jsonl"))
        ds = train.PretrainVideoDataset(rows, 16, use_face_detector=False, max_frames=3)
        full = train.PretrainVideoDataset(rows, 16, use_face_detector=False)[0]["video"]
        self.assertEqual(len(ds), 2)
        torch.testing.assert_close(torch.cat([ds[0]["video"], ds[1]["video"]]), full)
        batch = next(iter(DataLoader(ds, batch_size=2, collate_fn=train.collate_pretrain_videos)))
        self.assertEqual(batch["video_lengths"].tolist(), [3, 2])
        self.assertEqual(batch["video"].shape, (2, 3, 3, 16, 16))
        self.assertEqual(batch["video"][1, 2].abs().sum().item(), 0)

    def test_autocast_pretraining_backpropagates_through_final_representations(self):
        cfg = train.ModelConfig(vocab_size=6, conv3d_channels=8, d_video=8,
                                heads=2, video_layers=1, ff_mult=1, dropout=0)
        encoder = train.VideoEncoder(cfg).train()
        batch = {"video": torch.rand(2, 3, 3, 16, 16) * 2 - 1,
                 "video_lengths": torch.tensor([3, 1])}
        with torch.autocast("cpu", dtype=torch.bfloat16):
            batch = train.move_batch(batch, torch.device("cpu"))
            losses = train.visual_pretraining_losses(encoder, batch)
        for value in losses.values():
            self.assertEqual(value.dtype, torch.float32)
            self.assertTrue(torch.isfinite(value))
        losses["total"].backward()
        for parameter in encoder.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_pretraining_only_updates_video_and_resumes_epoch_count(self):
        args = self.args("--lambda-pretrain-temporal", "0.2", "--lambda-pretrain-augmentation", "3",
                         "--lambda-pretrain-variance", "4", "--lambda-pretrain-covariance", "0.5",
                         "--pretrain-variance-floor", "0.8", "--pretrain-original-probability", "0.65")
        initial = {}
        construct = train.UnnobaModel

        def capture_model(*positional, **kwargs):
            model = construct(*positional, **kwargs)
            initial.update({name: value.clone() for name, value in model.state_dict().items()})
            return model

        log = io.StringIO()
        with patch.object(train, "UnnobaModel", side_effect=capture_model), \
                patch.object(train, "visual_pretraining_losses", wraps=train.visual_pretraining_losses) as losses, \
                contextlib.redirect_stdout(log):
            train.train(args)
        self.assertEqual(losses.call_args.kwargs, {
            "lambda_temporal": 0.2, "lambda_augmentation": 3, "lambda_variance": 4,
            "lambda_covariance": 0.5, "variance_floor": 0.8, "original_probability": 0.65,
        })
        self.assertIn("variance=", log.getvalue())
        self.assertIn("covariance=", log.getvalue())
        saved = torch.load(args.output, weights_only=False)
        for key in ("lambda_pretrain_temporal", "lambda_pretrain_augmentation", "lambda_pretrain_variance",
                    "lambda_pretrain_covariance", "pretrain_variance_floor", "pretrain_original_probability"):
            self.assertEqual(saved["training_args"][key], getattr(args, key))
        self.assertEqual(saved["training_stage"], "pretrain")
        self.assertEqual(saved["pretrain_epoch"], 1)
        self.assertEqual(saved["epoch"], 0)
        self.assertFalse(saved["confidence"]["trained"])
        for prefix in ("video_encoder.front.", "video_encoder.frame_encoder.", "video_encoder.layers.", "video_encoder.out_norm."):
            self.assertTrue(any(not torch.equal(value, initial[name])
                                for name, value in saved["model_state"].items() if name.startswith(prefix)))
        for name, value in saved["model_state"].items():
            if not name.startswith("video_encoder."):
                torch.testing.assert_close(value, initial[name], atol=0, rtol=0)
        with contextlib.redirect_stdout(io.StringIO()):
            train.train(self.args("--resume", args.output))
        self.assertEqual(torch.load(args.output, weights_only=False)["pretrain_epoch"], 2)

    def test_pretraining_precedes_supervised_stages_and_is_optional_on_resume(self):
        args = self.args("--epochs", "1", "--confidence-epochs", "1")
        stages = []
        save = train.save_checkpoint

        def record(*positional, **kwargs):
            stages.append(kwargs.get("stage", "token"))
            return save(*positional, **kwargs)

        with patch.object(train, "save_checkpoint", side_effect=record), contextlib.redirect_stdout(io.StringIO()):
            train.train(args)
        self.assertEqual(stages, ["pretrain", "token", "confidence"])
        saved = torch.load(args.output, weights_only=False)
        self.assertEqual(saved["pretrain_epoch"], 1)
        self.assertEqual(saved["epoch"], 1)
        self.assertTrue(saved["confidence"]["trained"])
        resumed = self.args("--resume", args.output, "--epochs", "1")
        resumed.pretrain_visual_encoder = False
        resumed.pretrain_manifest = None
        with patch.object(train, "pretrain_visual_encoder") as pretrain, contextlib.redirect_stdout(io.StringIO()):
            train.train(resumed)
        pretrain.assert_not_called()
        saved = torch.load(args.output, weights_only=False)
        self.assertEqual(saved["pretrain_epoch"], 1)
        self.assertEqual(saved["epoch"], 2)

    def test_pretrain_only_overrides_supervised_epochs_and_skips_supervised_data_and_metrics(self):
        args = self.args("--pretrain-only", "--epochs", "3", "--confidence-epochs", "2",
                         "--metrics-csv", "auto", "--plot-learning-curves")
        args.pretrain_visual_encoder = False  # The new flag enables pretraining itself.
        with patch.object(train, "VideoTextWindowDataset", side_effect=AssertionError("supervised data loaded")), \
                patch.object(train, "evaluate_samples", side_effect=AssertionError("supervised metrics ran")), \
                contextlib.redirect_stdout(io.StringIO()):
            train.train(args)
        saved = torch.load(args.output, weights_only=False)
        self.assertEqual((saved["training_stage"], saved["epoch"], saved["confidence_epoch"]), ("pretrain", 0, 0))
        self.assertEqual(saved["pretrain_epoch"], 1)
        self.assertFalse(saved["tokenizer_pending"])
        self.assertFalse(Path(args.output).with_suffix(".metrics.csv").exists())
        self.assertFalse(Path(args.output).with_suffix(".learning.csv").exists())

    def test_unlabeled_pretrain_only_resumes_full_training_and_preserves_visual_weights(self):
        for fusion in ("residual", "frame"):
            with self.subTest(fusion=fusion):
                args = self.args("--pretrain-only", "--text-fusion", fusion)
                args.manifest = None
                with contextlib.redirect_stdout(io.StringIO()):
                    train.train(args)
                saved = torch.load(args.output, weights_only=False)
                self.assertTrue(saved["tokenizer_pending"])
                self.assertEqual(saved["tokenizer"]["itos"], train.CharTokenizer.build([]).itos)
                # Additional explicit pretraining preserves the pending vocabulary.
                args.resume = args.output
                with contextlib.redirect_stdout(io.StringIO()):
                    train.train(args)
                saved = torch.load(args.output, weights_only=False)
                self.assertEqual(saved["pretrain_epoch"], 2)
                self.assertTrue(saved["tokenizer_pending"])

                resumed = self.args("--resume", args.output, "--epochs", "1", "--confidence-epochs", "1")
                resumed.pretrain_visual_encoder = False
                resumed.pretrain_manifest = None
                resumed.epochs = 0
                with self.assertRaisesRegex(ValueError, "requires --epochs > 0"):
                    train.train(resumed)
                resumed.epochs = 1
                stages = []
                configure = train.configure_token_stage
                save = train.save_checkpoint

                def check_visual_weights(model):
                    for name, value in model.video_encoder.state_dict().items():
                        torch.testing.assert_close(value, saved["model_state"]["video_encoder." + name], rtol=0, atol=0)
                    return configure(model)

                def record(*positional, **kwargs):
                    stages.append(kwargs.get("stage", "token"))
                    return save(*positional, **kwargs)

                with patch.object(train, "pretrain_visual_encoder", side_effect=AssertionError("pretraining repeated")), \
                        patch.object(train, "configure_token_stage", side_effect=check_visual_weights), \
                        patch.object(train, "save_checkpoint", side_effect=record), \
                        contextlib.redirect_stdout(io.StringIO()):
                    train.train(resumed)
                self.assertEqual(stages, ["token", "confidence"])
                trained = torch.load(args.output, weights_only=False)
                self.assertFalse(trained["tokenizer_pending"])
                self.assertEqual(trained["tokenizer"]["itos"], train.CharTokenizer.build(["ab"]).itos)
                self.assertEqual((trained["pretrain_epoch"], trained["epoch"], trained["confidence_epoch"]), (2, 1, 1))
                self.assertTrue(trained["confidence"]["trained"])
                # Once supervised training fixes the vocabulary, mismatches still fail.
                mismatch = self.root / "mismatch.jsonl"
                mismatch.write_text(json.dumps({"video": "clip.avi", "text": "cd", "windows": [[0, 1], [2, 4]]}) + "\n")
                resumed.manifest = str(mismatch)
                with self.assertRaisesRegex(ValueError, "character vocabulary"):
                    train.train(resumed)

    def test_pretrain_only_requires_source_and_rejects_confidence_only(self):
        cases = [
            ({"pretrain_only": True, "pretrain_manifest": None}, "requires --pretrain-manifest"),
            ({"pretrain_only": True, "confidence_only": True}, "cannot be combined"),
            ({"manifest": None}, "--manifest is required"),
        ]
        for overrides, message in cases:
            args = self.args()
            for name, value in overrides.items():
                setattr(args, name, value)
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, message):
                train.train(args)

    def test_unlabeled_pretrain_only_keeps_pretrained_lm_and_resumes_offline(self):
        from tokenizers import Tokenizer, models
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

        lm_path = self.root / "tiny-lm"
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(models.WordLevel(
                {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "ab": 4}, unk_token="<unk>")),
            pad_token="<pad>", bos_token="<bos>", eos_token="<eos>", unk_token="<unk>",
        )
        tokenizer.save_pretrained(lm_path)
        lm = GPT2LMHeadModel(GPT2Config(
            vocab_size=len(tokenizer), n_embd=8, n_head=2, n_layer=1, n_positions=32,
            bos_token_id=1, eos_token_id=2, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
        ))
        lm.save_pretrained(lm_path)
        args = self.args("--pretrain-only", "--pretrained-lm", str(lm_path), "--pretrained-lm-local-files-only")
        args.manifest = None
        with contextlib.redirect_stdout(io.StringIO()):
            train.train(args)
        saved = torch.load(args.output, weights_only=False)
        self.assertFalse(saved["tokenizer_pending"])
        lm_path.rename(self.root / "hidden-lm")  # Resume must rebuild from the checkpoint.
        resumed = self.args("--resume", args.output, "--epochs", "1", "--confidence-epochs", "1")
        resumed.pretrain_visual_encoder = False
        resumed.pretrain_manifest = None
        with contextlib.redirect_stdout(io.StringIO()):
            train.train(resumed)
        trained = torch.load(args.output, weights_only=False)
        self.assertEqual(trained["tokenizer"], saved["tokenizer"])
        self.assertEqual((trained["pretrain_epoch"], trained["epoch"], trained["confidence_epoch"]), (1, 1, 1))
        self.assertTrue(trained["confidence"]["trained"])
        for name, value in saved["model_state"].items():
            if name.startswith("text_encoder."):
                torch.testing.assert_close(trained["model_state"][name], value, atol=0, rtol=0)

    def test_invalid_options_and_validation_overlap_fail_early(self):
        cases = [
            ({"pretrain_manifest": None}, "requires --pretrain-manifest"),
            ({"pretrain_visual_encoder": False}, "requires --pretrain-visual-encoder"),
            ({"pretrain_epochs": 0}, "--pretrain-epochs"),
            ({"pretrain_adjacent_frames": 1}, "--pretrain-adjacent-frames"),
            ({"confidence_only": True}, "cannot be combined"),
            ({"output": str(self.root / "pretrain.jsonl")}, "must differ"),
        ]
        for overrides, message in cases:
            args = self.args()
            for name, value in overrides.items():
                setattr(args, name, value)
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, message):
                train.train(args)
        # Supervised training is disjoint, but unlabeled pretraining leaks a
        # validation video. Reject it before decoding or model construction.
        valid = self.root / "valid.jsonl"
        valid.write_text(json.dumps({"video": "held-out.avi", "text": "ab", "windows": []}) + "\n")
        (self.root / "pretrain.jsonl").write_text(json.dumps({"video": "held-out.avi"}) + "\n")
        with self.assertRaisesRegex(ValueError, "overlap"):
            train.train(self.args("--validation-manifest", str(valid)))

    def test_loss_parameters_reject_nonfinite_negative_weights_and_nonpositive_floor(self):
        for key in ("lambda_pretrain_temporal", "lambda_pretrain_augmentation", "lambda_pretrain_variance",
                    "lambda_pretrain_covariance", "pretrain_variance_floor", "pretrain_original_probability"):
            invalid = [-1, float("nan"), float("inf"), float("-inf")]
            if key == "pretrain_variance_floor":
                invalid.append(0)
            if key == "pretrain_original_probability":
                invalid.append(1.1)
            for value in invalid:
                with self.subTest(key=key, value=value):
                    args = self.args()
                    setattr(args, key, value)
                    with self.assertRaisesRegex(ValueError, "--" + key.replace("_", "-")):
                        train.train(args)

    def test_default_objective_and_zero_weight_ablation(self):
        args = self.args()
        self.assertEqual(args.pretrain_original_probability, 0.2)
        self.assertEqual((args.lambda_pretrain_temporal, args.lambda_pretrain_augmentation,
                          args.lambda_pretrain_variance, args.lambda_pretrain_covariance,
                          args.pretrain_variance_floor), (0, 1, 1, 0.04, 1))
        for key in ("lambda_pretrain_temporal", "lambda_pretrain_augmentation", "lambda_pretrain_variance",
                    "lambda_pretrain_covariance"):
            setattr(args, key, 0)
        train.validate_pretraining_args(args)
        for probability in (0, 1):
            args.pretrain_original_probability = probability
            train.validate_pretraining_args(args)


if __name__ == "__main__":
    unittest.main()
