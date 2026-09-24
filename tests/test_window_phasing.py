"""Relative window phasing, separate causal contexts, and training integration."""

import copy
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

import run_all_variants as runner
import train


class WindowPhasingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(11)
        self.tokenizer = train.CharTokenizer.build(['abbc'])
        self.row = {'video': 'unused.avi', 'text': 'abbc',
                    'windows': [[0, 2], [5, 14], [15, 17], [18, 18]]}

    def samples(self, fraction=0.0, max_frames=None):
        video = torch.arange(19 * 3 * 16 * 16, dtype=torch.float32).reshape(19, 3, 16, 16) / 20000

        def read(*args, start_frame=0, max_frames=None, **kwargs):
            end = len(video) if max_frames is None else start_frame + max_frames
            return video[start_frame:end], 25.0

        with patch.object(train, 'count_video_frames', return_value=19), \
                patch.object(train, 'read_video_mouth_tensor', side_effect=read):
            dataset = train.VideoTextWindowDataset(
                [self.row], self.tokenizer, 16, use_face_detector=False,
                max_frames=max_frames, window_phasing=fraction)
            return [dataset[i] for i in range(len(dataset))]

    def test_floor_endpoint_first_window_gaps_and_repeated_tokens(self):
        pad = self.tokenizer.pad_id
        a, b, c = self.tokenizer.encode('abc')
        for fraction, expected in (
            (0.0, [pad] * 19),
            (0.1, [pad] * 5 + [a] + [pad] * 13),
            (0.5, [pad] * 5 + [a] * 5 + [pad] * 5 + [b] + [pad] * 3),
            (1.0, [pad] * 5 + [a] * 10 + [b] * 4),
        ):
            with self.subTest(fraction=fraction):
                sample = self.samples(fraction)[0]
                self.assertEqual(sample['frame_previous_targets'].tolist(), expected)
                self.assertEqual(sample['frame_targets'].tolist(), [a] * 3 + [pad] * 2 + [b] * 13 + [c])
                self.assertEqual(sample['frame_windows'].tolist(), self.row['windows'])
                self.assertEqual(sample['text_available_at'].tolist(), [0, 3, 15, 18, 19])

    def test_original_window_phase_survives_section_cuts_and_batch_padding(self):
        full = self.samples(0.5)[0]
        sections = self.samples(0.5, max_frames=4)
        for name in ('frame_targets', 'frame_previous_targets'):
            torch.testing.assert_close(torch.cat([s[name] for s in sections]), full[name])
        batch = train.make_collate(self.tokenizer.pad_id)(sections)
        self.assertEqual(batch['frame_previous_targets'].shape, (5, 4))
        self.assertEqual(batch['frame_previous_targets'][-1, -1].item(), self.tokenizer.pad_id)
        self.assertEqual(batch['frame_previous_targets'][2].tolist(),
                         [self.tokenizer.encode('a')[0]] * 2 + [self.tokenizer.pad_id] * 2)

    def reference_loss(self, logits, targets, weights):
        valid = targets != 0
        current = F.one_hot(targets[valid], logits.shape[-1]).to(logits.dtype)
        return -((current * logits[valid].log_softmax(-1)).sum(-1) * weights[valid]).mean()

    def model(self, **kwargs):
        return train.UnnobaModel(train.ModelConfig(
            vocab_size=self.tokenizer.vocab_size, mouth_size=16, conv3d_channels=8,
            d_video=8, d_text=8, d_fusion=8, heads=2, video_layers=1,
            text_layers=1, ff_mult=1, text_fusion='frame', **kwargs))

    def args(self):
        args = train.build_argparser().parse_args([
            '--manifest', 'unused', '--window-phasing', '0.5'])
        args.pad_id = self.tokenizer.pad_id
        return args

    def reference_phased_loss(self, model, batch, out):
        # Brute-force oracle: recompute the whole fusion branch with a second
        # mask, including non-phased frames. Shifting the availability mask
        # drops its last visible state, independently of the sparse chunk code.
        _, allowed = model.build_confidence_masks(
            batch['video_lengths'], batch['text_lengths'], batch['text_available_at'],
            batch['video'].shape[1], batch['teacher'].shape[1])
        previous_allowed = torch.zeros_like(allowed)
        previous_allowed[:, :, :-1] = allowed[:, :, 1:]
        previous_logits = model.mixed.forward_frame(
            out['video_encoded'], out['text_encoded'], previous_allowed)['token_logits']
        pad = self.tokenizer.pad_id
        targets, previous = batch['frame_targets'], batch['frame_previous_targets']
        valid = targets != pad
        phased = valid & (previous != pad)
        current_ce = F.cross_entropy(out['token_logits'].transpose(1, 2), targets,
                                     ignore_index=pad, reduction='none')
        previous_ce = F.cross_entropy(previous_logits.transpose(1, 2), previous,
                                      ignore_index=pad, reduction='none')
        return torch.where(phased, (current_ce + previous_ce) * 0.5, current_ce)[valid].mean()

    def test_position_weights_preserve_global_frame_mean_and_gradients(self):
        logits = torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True)
        reference_logits = logits.detach().clone().requires_grad_()
        targets = torch.tensor([[1, 2, 3, 4], [2, 0, 0, 0]])
        weights = torch.tensor([[1, .5, .5, .5], [.5, .5, 1, 1]])
        actual = train.token_cross_entropy(logits, targets, 0, weights)
        expected = self.reference_loss(reference_logits, targets, weights)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        expected.backward()
        torch.testing.assert_close(logits.grad, reference_logits.grad)
        self.assertEqual(logits.grad[1, 1:].abs().sum().item(), 0)

    def test_disabled_phasing_and_all_ignored_targets(self):
        logits = torch.randn(2, 3, 5, requires_grad=True)
        targets = torch.tensor([[1, 2, 0], [3, 0, 0]])
        expected = F.cross_entropy(logits.reshape(-1, 5), targets.reshape(-1), ignore_index=0)
        torch.testing.assert_close(train.token_cross_entropy(logits, targets, 0), expected, atol=0, rtol=0)
        torch.testing.assert_close(train.token_cross_entropy(logits, targets, 0, torch.ones_like(targets)), expected)
        empty = train.token_cross_entropy(logits, torch.zeros_like(targets), 0, torch.ones_like(targets))
        empty.backward()
        self.assertEqual(empty.item(), 0)
        self.assertEqual(logits.grad.abs().sum().item(), 0)

    def test_chunked_amp_matches_fp32_weighted_targets_and_backward(self):
        devices = [('cpu', torch.bfloat16)]
        if torch.cuda.is_available():
            devices.append(('cuda', torch.float16))
        for device, dtype in devices:
            with self.subTest(device=device):
                logits = torch.randn(2, 200, 7, device=device, dtype=dtype, requires_grad=True)
                targets = torch.randint(1, 7, (2, 200), device=device)
                weights = torch.randint(1, 3, (2, 200), device=device).float() / 2
                targets.reshape(-1)[128:256] = 0  # Entire chunk ignored.
                targets[1, -17:] = 0
                reference_logits = logits.detach().float().requires_grad_()
                expected = self.reference_loss(reference_logits, targets, weights)
                with torch.amp.autocast(device, dtype=dtype):
                    actual = train.token_cross_entropy(logits, targets, 0, weights)
                    with torch.no_grad():
                        inference_loss = train.token_cross_entropy(logits, targets, 0, weights)
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(inference_loss, expected)
                actual.backward()
                expected.backward()
                torch.testing.assert_close(logits.grad, reference_logits.grad.to(dtype), atol=2e-5, rtol=1e-2)
                self.assertTrue(torch.isfinite(logits.grad).all())

    def test_separate_context_loss_and_gradients_match_dense_reference(self):
        for max_frames in (None, 4):
            for text_weight in (0.0, 1.0):
                with self.subTest(max_frames=max_frames, text_weight=text_weight):
                    batch = train.make_collate(self.tokenizer.pad_id)(self.samples(0.5, max_frames))
                    model = self.model(dropout=0, frame_vcross_weight=text_weight).eval()
                    reference = copy.deepcopy(model)
                    actual = train.compute_losses(model, batch, model(batch), self.args())['token']
                    expected = self.reference_phased_loss(reference, batch, reference(batch))
                    torch.testing.assert_close(actual, expected)
                    with torch.no_grad():
                        evaluated = train.compute_losses(model, batch, model(batch), self.args())['token']
                    torch.testing.assert_close(evaluated, expected)
                    actual.backward()
                    expected.backward()
                    for (name, param), (_, ref) in zip(model.named_parameters(), reference.named_parameters()):
                        if ref.grad is None:
                            self.assertIsNone(param.grad, name)
                        else:
                            torch.testing.assert_close(param.grad, ref.grad, atol=2e-6, rtol=2e-4, msg=name)

    def test_checkpoint_recomputation_preserves_dropout_gradients(self):
        batch = train.make_collate(self.tokenizer.pad_id)(self.samples(0.5, 4))
        model = self.model(dropout=0.2).train()
        reference = copy.deepcopy(model)
        torch.manual_seed(31)
        actual = train.compute_losses(model, batch, model(batch), self.args())['token']
        torch.manual_seed(31)
        with patch.object(train, 'checkpoint', side_effect=lambda fn, *args, **kwargs: fn(*args)):
            expected = train.compute_losses(reference, batch, reference(batch), self.args())['token']
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        actual.backward()
        expected.backward()
        for (name, param), (_, ref) in zip(model.named_parameters(), reference.named_parameters()):
            if ref.grad is not None:
                torch.testing.assert_close(param.grad, ref.grad, atol=0, rtol=0, msg=name)

    def test_previous_target_cannot_see_itself_or_later_text(self):
        batch = train.make_collate(self.tokenizer.pad_id)(self.samples(0.5))
        # The second b window: both targets are b, but only the current path
        # may see the first b. The previous path must stop at the earlier a.
        frame = 15
        previous = batch['frame_previous_targets'][0, frame].item()
        batch['frame_previous_targets'].fill_(self.tokenizer.pad_id)
        batch['frame_previous_targets'][0, frame] = previous
        model = self.model(dropout=0).eval()
        with torch.inference_mode():
            original = model(batch)
            original_loss = train.previous_frame_cross_entropy(model, batch, original, self.tokenizer.pad_id)
            for teacher_index in (1, 2, 3, 4):
                changed = dict(batch)
                changed['teacher'] = batch['teacher'].clone()
                changed['teacher'][0, teacher_index] = self.tokenizer.unk_id
                out = model(changed)
                loss = train.previous_frame_cross_entropy(model, changed, out, self.tokenizer.pad_id)
                if teacher_index == 1:
                    self.assertGreater(abs((loss - original_loss).item()), 1e-6)
                else:
                    torch.testing.assert_close(loss, original_loss, atol=0, rtol=0)
                if teacher_index == 2:
                    self.assertGreater((out['token_logits'][0, frame] - original['token_logits'][0, frame]).abs().max().item(), 1e-6)
                elif teacher_index >= 3:
                    torch.testing.assert_close(out['token_logits'][0, frame], original['token_logits'][0, frame], atol=0, rtol=0)

    def test_disabled_or_empty_phasing_skips_extra_branch(self):
        model = self.model(dropout=0).eval()
        for fraction, enabled in ((0, True), (0.5, False)):
            batch = train.make_collate(self.tokenizer.pad_id)(self.samples(fraction))
            args = self.args()
            if not enabled:
                args.window_phasing = 0
            out = model(batch)
            with patch.object(train, 'previous_frame_cross_entropy', side_effect=AssertionError('extra branch')):
                actual = train.compute_losses(model, batch, out, args)['token']
            expected = train.token_cross_entropy(out['token_logits'], batch['frame_targets'], args.pad_id)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_previous_branch_reuses_projections_and_bounds_chunks_under_amp(self):
        model = self.model(dropout=0)
        a, b = self.tokenizer.encode('ab')
        # 259 phased frames force three chunks, including a partial final one.
        batch = {
            'frame_targets': torch.tensor([[a] + [b] * 259]),
            'frame_previous_targets': torch.tensor([[0] + [a] * 259]),
            'text_lengths': torch.tensor([3]),
            'text_available_at': torch.tensor([[0, 1, 260]]),
        }
        video = torch.randn(1, 260, 8, requires_grad=True)
        text = torch.randn(1, 3, 8, requires_grad=True)
        allowed = batch['text_available_at'][:, None] <= torch.arange(260)[None, :, None]
        with torch.amp.autocast('cpu', dtype=torch.bfloat16):
            out = model.mixed.forward_frame(video, text, allowed)
            with patch.object(model.mixed.video_cross, 'project_kv', side_effect=AssertionError('reprojected text')), \
                    patch.object(model.mixed.video_cross.q_proj, 'forward', side_effect=AssertionError('reprojected queries')), \
                    patch.object(model.mixed, 'frame_logits', wraps=model.mixed.frame_logits) as head:
                loss = train.previous_frame_cross_entropy(model, batch, out, 0)
                self.assertEqual([call.args[0].shape[1] for call in head.call_args_list], [128, 128, 3])
                loss.backward()
                # Backpropagation recomputes bounded chunks, never all frames.
                self.assertGreater(head.call_count, 3)
                self.assertTrue(all(call.args[0].shape[1] <= 128 for call in head.call_args_list))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(video.grad).all())
        self.assertTrue(torch.isfinite(text.grad).all())
        # Only BOS is visible to this previous-token path.
        self.assertGreater(text.grad[:, :1].abs().sum().item(), 0)
        self.assertEqual(text.grad[:, 1:].abs().sum().item(), 0)

    def test_invalid_fractions_and_non_frame_training_are_rejected(self):
        parser = train.build_argparser()
        for value in (-0.1, 1.01, float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'window-phasing'):
                    train.VideoTextWindowDataset([], self.tokenizer, 16, window_phasing=value)
                with self.assertRaisesRegex(ValueError, 'window-phasing'):
                    train.train(parser.parse_args(['--manifest', 'unused', f'--window-phasing={value}']))
        with patch.object(train, 'load_manifest', return_value=[self.row]):
            with self.assertRaisesRegex(ValueError, 'requires --text-fusion frame'):
                train.train(parser.parse_args(['--manifest', 'unused', '--window-phasing', '0.1']))

    def test_runner_routes_phasing_only_to_variant_five_and_validates_overrides(self):
        parser = runner.build_argparser()
        args = parser.parse_args(['-N', '1', '--window-phasing', '0.1',
                                  '--variant-args', '5:--window-phasing 0.25'])
        _, _, _, stages = runner.build_commands(parser, args, Path('unused-run'))
        self.assertFalse(any('window-phasing' in flag for flag in stages[0]['command']))
        for index, stage in enumerate(stages[1::2], 1):
            parsed = runner.VARIANTS[index - 1].build_argparser().parse_args(stage['command'][2:])
            self.assertEqual(parsed.window_phasing, 0.25 if index == 5 else 0)
        for flags in (['--window-phasing', 'nan'], ['--window-phasing', '-0.1'],
                      ['--window-phasing', '1.1'], ['--variant-args', '1:--window-phasing 0.1']):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    runner.build_commands(parser, parser.parse_args(['-N', '1', *flags]), Path('unused-run'))

    def test_training_uses_phased_loss_and_saves_setting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / 'data.jsonl'
            manifest.write_text(json.dumps(self.row) + '\n')
            checkpoint = root / 'model.pt'
            args = train.build_argparser().parse_args([
                '--manifest', str(manifest), '--output', str(checkpoint), '--device', 'cpu',
                '--epochs', '1', '--confidence-epochs', '0', '--batch-size', '1', '--log-every', '0',
                '--no-plot-learning-curves',  # This loss-only fixture mocks decoded video tensors.
                '--no-face-detector', '--mouth-size', '16', '--conv3d-channels', '8',
                '--d-video', '8', '--d-text', '8', '--d-fusion', '8', '--heads', '2',
                '--video-layers', '1', '--text-layers', '1', '--ff-mult', '1', '--dropout', '0',
                '--text-fusion', 'frame', '--window-phasing', '0.5'])
            compute = train.compute_losses
            checked = []

            def check_loss(model, batch, out, args):
                self.assertTrue((batch['frame_previous_targets'] != self.tokenizer.pad_id).any())
                losses = compute(model, batch, out, args)
                expected = self.reference_phased_loss(model, batch, out)
                torch.testing.assert_close(losses['token'], expected)
                checked.append(True)
                return losses

            with contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(train, 'read_video_mouth_tensor', return_value=(torch.randn(19, 3, 16, 16), 25)), \
                    patch.object(train, 'compute_losses', side_effect=check_loss):
                train.train(args)
            self.assertEqual(checked, [True])
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            self.assertEqual(saved['training_args']['window_phasing'], 0.5)


if __name__ == '__main__':
    unittest.main()
