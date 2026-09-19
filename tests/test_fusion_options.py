"""Fusion ablations: dimensions, gradients, streaming, and checkpoint compatibility."""

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

import infer
import train


class FusionOptionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(7)
        self.tokenizer = train.CharTokenizer.build(['ab'])

    def config(self, **overrides):
        values = dict(vocab_size=self.tokenizer.vocab_size, mouth_size=16,
                      conv3d_channels=8, d_video=8, d_text=12, d_fusion=16,
                      heads=2, video_layers=1, text_layers=1, ff_mult=1,
                      conv_kernel=3, dropout=0, text_fusion='frame')
        values.update(overrides)
        return train.ModelConfig(**values)

    def batch(self):
        a, b = self.tokenizer.encode('ab')
        return train.make_collate(self.tokenizer.pad_id)([{
            'video': torch.randn(6, 3, 16, 16), 'teacher': torch.tensor([self.tokenizer.bos_id, a, b]),
            'targets': torch.tensor([a, b, self.tokenizer.eos_id]),
            'token_video_end': torch.tensor([2, 5, 5]),
            'text_available_at': torch.tensor([0, 3, 6]),
            'windows': torch.tensor([[0, 2], [3, 5], [5, 5]]), 'window_count': 2,
            'frame_targets': torch.tensor([a, a, a, b, b, b]),
            'frame_windows': torch.tensor([[0, 2], [3, 5]]),
            'text': 'ab', 'path': 'unused',
        }])

    def test_unprojected_video_sets_all_fusion_widths(self):
        for fusion in ('frame', 'residual', 'cross_only', 'weighted'):
            with self.subTest(fusion=fusion):
                model = train.UnnobaModel(self.config(use_video_projection=False, text_fusion=fusion)).eval()
                self.assertIsInstance(model.mixed.vproj, torch.nn.Identity)
                self.assertFalse(any(name.startswith('mixed.vproj.') for name in model.state_dict()))
                self.assertEqual(model.mixed.tproj.weight.shape, (8, 12))
                batch = self.batch()
                out = model(batch)
                self.assertEqual(out['video_projected'].shape[-1], 8)
                self.assertEqual(out['text_projected'].shape[-1], 8)
                self.assertTrue(torch.equal(out['video_projected'], out['video_encoded']))
                self.assertEqual(out['token_logits'].shape, (1, 6 if fusion == 'frame' else 3, self.tokenizer.vocab_size))
                self.assertEqual(model.forward_confidence(batch).shape, (1, 6))

    def test_frame_training_confidence_targets_and_streaming_agree(self):
        for projected in (True, False):
            for weight in (0.0, 0.25, 1.0):
                with self.subTest(projected=projected, weight=weight), torch.inference_mode():
                    model = train.UnnobaModel(self.config(use_video_projection=projected, frame_vcross_weight=weight)).eval()
                    batch = self.batch()
                    out = model(batch)
                    target_logits = train.frozen_next_token_logits_per_frame(
                        model, out['video_encoded'], out['text_encoded'], batch)
                    torch.testing.assert_close(target_logits, out['token_logits'])
                    confidence = model.forward_confidence(batch)
                    state = model.init_stream_state()
                    logits, scores = [], []
                    for frame in range(6):
                        for token, available in enumerate(batch['text_available_at'][0]):
                            if int(available) == frame:
                                state = model.stream_push_text_token(batch['teacher'][:, token], state)
                        score, state = model.stream_push_video_frame(batch['video'][:, frame], state)
                        logit, _ = model.stream_predict_next(state)
                        logits.append(logit)
                        scores.append(score)
                    torch.testing.assert_close(torch.cat(logits, dim=1), out['token_logits'], atol=2e-5, rtol=2e-5)
                    torch.testing.assert_close(torch.cat(scores, dim=1), confidence, atol=2e-5, rtol=2e-5)

    def test_zero_weight_removes_text_dependence_and_text_gradients(self):
        for weight in (0.0, 0.3):
            with self.subTest(weight=weight):
                block = train.MixedBlock(self.config(use_video_projection=False, frame_vcross_weight=weight))
                video = torch.randn(1, 4, 8, requires_grad=True)
                text = torch.randn(1, 3, 12, requires_grad=True)
                allowed = torch.ones(1, 4, 3, dtype=torch.bool)
                logits = block.forward_frame(video, text, allowed)['token_logits']
                changed = block.forward_frame(video, text * -7, allowed)['token_logits']
                logits.square().mean().backward()
                self.assertGreater(video.grad.abs().sum().item(), 0)
                if weight == 0:
                    torch.testing.assert_close(logits, changed, atol=0, rtol=0)
                    self.assertEqual(text.grad.abs().sum().item(), 0)
                else:
                    self.assertFalse(torch.allclose(logits, changed))
                    self.assertGreater(text.grad.abs().sum().item(), 0)

    def test_identity_video_projection_has_no_projection_penalties(self):
        model = train.UnnobaModel(self.config(use_video_projection=False))
        batch = self.batch()
        args = train.build_argparser().parse_args(['--manifest', 'unused', '--lambda-vproj', '1', '--lambda-vnorm', '1'])
        args.pad_id = self.tokenizer.pad_id
        losses = train.compute_losses(model, batch, model(batch), args)
        self.assertEqual(losses['vproj'].item(), 0)
        self.assertEqual(losses['vnorm'].item(), 0)
        losses['total'].backward()
        self.assertGreater(model.video_encoder.frame_encoder.proj.weight.grad.abs().sum().item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA AMP requires a GPU')
    def test_unprojected_frame_token_and_confidence_gradients_under_amp(self):
        model = train.UnnobaModel(self.config(use_video_projection=False, frame_vcross_weight=.25)).cuda()
        batch = train.move_batch(self.batch(), torch.device('cuda'))
        args = train.build_argparser().parse_args(['--manifest', 'unused'])
        args.pad_id = self.tokenizer.pad_id
        with torch.amp.autocast('cuda'):
            out = model(batch)
            losses = train.compute_losses(model, batch, out, args)
        losses['total'].backward()
        self.assertTrue(torch.isfinite(losses['total']))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        model.zero_grad(set_to_none=True)
        train.configure_confidence_stage(model)
        train.set_confidence_train_mode(model)
        with torch.amp.autocast('cuda'):
            loss, stats = train.document_confidence_loss(model, batch, args)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(stats['C']).all())
        self.assertGreater(model.mixed.conf_head.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(model.mixed.video_cross.q_proj.weight.grad)
        self.assertIsNone(model.mixed.token_head.weight.grad)

    def test_old_checkpoint_keeps_original_architecture_and_logits(self):
        model = train.UnnobaModel(self.config()).eval()
        config = asdict(model.cfg)
        del config['use_video_projection'], config['frame_vcross_weight']
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'old.pt'
            torch.save({'model_config': config, 'model_state': model.state_dict(),
                        'tokenizer': self.tokenizer.state_dict()}, path)
            restored, _, _, _ = infer.load_model(str(path), torch.device('cpu'))
        self.assertIsInstance(restored.mixed.vproj, torch.nn.Linear)
        self.assertEqual(restored.cfg.frame_vcross_weight, 1)
        batch = self.batch()
        torch.testing.assert_close(restored(batch)['token_logits'], model(batch)['token_logits'], atol=0, rtol=0)

    def test_train_save_infer_and_confidence_resume_restore_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / 'clip.avi'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 25, (32, 32))
            self.assertTrue(writer.isOpened())
            try:
                for value in range(6):
                    writer.write(np.full((32, 32, 3), value * 30, dtype=np.uint8))
            finally:
                writer.release()
            manifest = root / 'data.jsonl'
            manifest.write_text(json.dumps({'video': str(video), 'text': 'ab', 'windows': [[0, 2], [3, 5]]}) + '\n')
            checkpoint = root / 'model.pt'
            parser = train.build_argparser()
            flags = ['--manifest', str(manifest), '--output', str(checkpoint), '--device', 'cpu',
                     '--epochs', '1', '--confidence-epochs', '1', '--batch-size', '1', '--log-every', '0',
                     '--no-face-detector', '--mouth-size', '16', '--conv3d-channels', '8',
                     '--d-video', '8', '--d-text', '12', '--d-fusion', '16', '--heads', '2',
                     '--video-layers', '1', '--text-layers', '1', '--ff-mult', '1', '--dropout', '0',
                     '--text-fusion', 'frame', '--no-vproj', '--vcross-weight', '0.25']
            with contextlib.redirect_stdout(io.StringIO()):
                train.train(parser.parse_args(flags))
            restored, _, _, saved = infer.load_model(str(checkpoint), torch.device('cpu'))
            self.assertFalse(restored.cfg.use_video_projection)
            self.assertEqual(restored.cfg.d_fusion, 8)
            self.assertEqual(restored.cfg.frame_vcross_weight, .25)
            self.assertTrue(saved['confidence']['trained'])
            resume = ['--manifest', str(manifest), '--resume', str(checkpoint), '--output', str(root / 'resumed.pt'),
                      '--confidence-only', '--confidence-epochs', '1', '--device', 'cpu', '--log-every', '0', '--no-face-detector']
            with contextlib.redirect_stdout(io.StringIO()):
                train.train(parser.parse_args(resume))
            resumed = torch.load(root / 'resumed.pt', map_location='cpu', weights_only=False)
            self.assertEqual(resumed['model_config'], saved['model_config'])
            self.assertEqual(resumed['confidence_epoch'], 2)
            for weight in ('0', '1'):
                with self.assertRaisesRegex(ValueError, 'vcross weight'):
                    train.train(parser.parse_args([*resume, '--vcross-weight', weight]))
            saved['model_config']['use_video_projection'] = True
            torch.save(saved, root / 'projected.pt')
            with self.assertRaisesRegex(ValueError, 'cannot resume'):
                train.train(parser.parse_args([*resume, '--resume', str(root / 'projected.pt'), '--no-vproj']))

    def test_nonfinite_weights_rejected_before_reading_data(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'frame_vcross_weight must be finite'):
                    train.MixedBlock(self.config(frame_vcross_weight=value))
                args = train.build_argparser().parse_args(['--manifest', 'unused', f'--vcross-weight={value}'])
                with self.assertRaisesRegex(ValueError, '--vcross-weight must be finite'):
                    train.train(args)

    def test_inference_restores_amp_for_bos_frames_and_committed_text(self):
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'clip.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 25, (32, 32))
            self.assertTrue(writer.isOpened())
            try:
                for _ in range(3):
                    writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
            finally:
                writer.release()
            for device in devices:
                for saved, flags, requested in [(True, [], True), (False, [], False),
                                                 (None, [], False), (True, ['--no-amp'], False),
                                                 (False, ['--amp'], True)]:
                    with self.subTest(device=device, saved=saved, flags=flags):
                        model = train.UnnobaModel(self.config()).to(device).eval()
                        with torch.no_grad():
                            model.mixed.token_head.weight.zero_()
                            model.mixed.token_head.bias.zero_()
                            model.mixed.token_head.bias[self.tokenizer.encode('a')[0]] = 10
                        ckpt = {'confidence': {'trained': True, 'target': 'document_entropy_future_instability'}}
                        if saved is not None:
                            ckpt['training_args'] = {'amp': saved}
                        calls = []
                        push_text = model.stream_push_text_token
                        push_video = model.stream_push_video_frame

                        def text_step(token, state):
                            calls.append(('text', torch.is_autocast_enabled(device)))
                            return push_text(token, state)

                        def video_step(frame, state):
                            calls.append(('video', torch.is_autocast_enabled(device)))
                            return push_video(frame, state)

                        args = infer.build_argparser().parse_args([
                            '--checkpoint', 'unused', '--video', str(path), '--device', device,
                            '--json-events', '--max-tokens', '1', '--warmup-frames', '1',
                            '--min-frames-per-token', '0', '--confidence-threshold', '0',
                            '--confidence-min-threshold', '0', *flags])
                        output = io.StringIO()
                        with patch.object(infer, 'load_model', return_value=(model, self.tokenizer,
                                         {'mouth_size':16, 'use_face_detector':False}, ckpt)), \
                                patch.object(model, 'stream_push_text_token', side_effect=text_step), \
                                patch.object(model, 'stream_push_video_frame', side_effect=video_step), \
                                contextlib.redirect_stdout(output):
                            infer.main(args)
                        enabled = requested and device == 'cuda'
                        self.assertEqual(calls, [('text', enabled), ('video', enabled), ('text', enabled)])
                        events = [json.loads(line) for line in output.getvalue().splitlines()]
                        self.assertEqual(events[0]['amp'], enabled)
                        self.assertEqual(events[-1]['text'], 'a')


if __name__ == '__main__':
    unittest.main()
