"""Masked self-supervision, stochastic views, and pretraining-stage integration."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
        args = self.args()
        initial = {}
        construct = train.UnnobaModel

        def capture_model(*positional, **kwargs):
            model = construct(*positional, **kwargs)
            initial.update({name: value.clone() for name, value in model.state_dict().items()})
            return model

        with patch.object(train, "UnnobaModel", side_effect=capture_model), contextlib.redirect_stdout(io.StringIO()):
            train.train(args)
        saved = torch.load(args.output, weights_only=False)
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


if __name__ == "__main__":
    unittest.main()
