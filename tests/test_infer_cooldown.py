"""Regression coverage for per-token cooldowns in the streaming inference loop."""

import contextlib
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import infer
from train import CharTokenizer


class InferCooldownTests(unittest.TestCase):
    def run_stream(self, predictions, *, flags=(), confidences=None, flush=()):
        tokenizer = CharTokenizer.build(['abc'])
        args = infer.build_argparser().parse_args([
            '--checkpoint', 'unused', '--video', 'unused', '--device', 'cpu',
            '--json-events', '--warmup-frames', '1', '--min-frames-per-token', '0',
            '--confidence-relax-per-frame', '0', *flags,
        ])
        model = Mock()
        model.cfg = SimpleNamespace(mouth_size=16, text_fusion='residual')
        state = object()
        model.init_stream_state.return_value = state
        model.stream_push_text_token.return_value = state
        model.stream_push_video_frame.side_effect = [
            (torch.logit(torch.tensor([[confidence]])), state)
            for confidence in (confidences if confidences is not None else [0.99] * len(predictions))
        ]
        logits = []
        for token in [*predictions, *flush]:
            token_id = tokenizer.eos_id if token == '<eos>' else tokenizer.encode(token)[0]
            scores = torch.full((1, 1, tokenizer.vocab_size), -10.0)
            # A valid runner-up makes accidental fallback emission observable.
            scores[..., tokenizer.encode('c')[0]] = 8.0
            scores[..., token_id] = 10.0
            logits.append((scores, None))
        model.stream_predict_next.side_effect = logits
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 25.0
        capture.read.side_effect = [(True, object()) for _ in predictions] + [(False, None)]
        checkpoint = {'confidence': {'trained': True, 'target': 'document_entropy_future_instability'}}
        output = io.StringIO()
        with patch.object(infer, 'load_model', return_value=(model, tokenizer, {}, checkpoint)), \
                patch.object(infer.cv2, 'VideoCapture', return_value=capture), \
                patch.object(infer, 'FixedMouthCropper', return_value=Mock(return_value=torch.zeros(3, 16, 16))), \
                contextlib.redirect_stdout(output):
            infer.main(args)
        capture.release.assert_called_once()
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        tokens = [event for event in events if event['event'] in ('token', 'flush_token')]
        pushed = [int(call.args[0].item()) for call in model.stream_push_text_token.call_args_list]
        self.assertEqual(pushed, [tokenizer.bos_id, *[event['token_id'] for event in tokens]])
        self.assertEqual(model.stream_push_video_frame.call_count, events[-1]['frames'])
        return tokens, events

    def test_default_cooldown_skips_repeats_until_exact_boundary(self):
        tokens, events = self.run_stream('a' * 11)
        self.assertEqual([(event['frame'], event['token']) for event in tokens],
                         [(0, 'a'), (5, 'a'), (10, 'a')])
        self.assertEqual(events[0]['repeat_token_cooldown_frames'], 5)
        self.assertEqual(events[-1]['frames'], 11)

    def test_cooldown_remembers_each_token_across_other_emissions(self):
        tokens, _ = self.run_stream('abaab', flags=['--repeat-token-cooldown-frames', '3'])
        self.assertEqual([(event['frame'], event['token']) for event in tokens],
                         [(0, 'a'), (1, 'b'), (3, 'a'), (4, 'b')])

    def test_zero_disables_repeat_guard(self):
        tokens, _ = self.run_stream('aaaa', flags=['--repeat-token-cooldown-frames', '0'])
        self.assertEqual([event['frame'] for event in tokens], [0, 1, 2, 3])

    def test_skipped_frames_do_not_accumulate_stability(self):
        tokens, _ = self.run_stream('aaaaaaa', flags=[
            '--repeat-token-cooldown-frames', '3', '--stable-frames', '2'])
        self.assertEqual([event['frame'] for event in tokens], [1, 5])

    def test_skipped_confidence_peak_cannot_trigger_later_emission(self):
        tokens, _ = self.run_stream('aaaaa', flags=['--repeat-token-cooldown-frames', '3'],
                                    confidences=[0.99, 0.99, 0.99, 0.1, 0.99])
        self.assertEqual([event['frame'] for event in tokens], [0, 4])
        self.assertEqual(tokens[-1]['confidence_frame'], 4)

    def test_general_gap_still_applies_after_cooldown(self):
        tokens, _ = self.run_stream('aaaaaaa', flags=[
            '--repeat-token-cooldown-frames', '2', '--min-frames-per-token', '4'])
        self.assertEqual([event['frame'] for event in tokens], [3])

    def test_flush_respects_stream_and_flush_emission_cooldowns(self):
        for cooldown, expected in [(5, 'ab'), (0, 'abaa')]:
            with self.subTest(cooldown=cooldown):
                tokens, _ = self.run_stream('a', flags=[
                    '--repeat-token-cooldown-frames', str(cooldown), '--flush-tokens', '3'],
                    flush='baa')
                self.assertEqual(''.join(event['token'] for event in tokens), expected)
        tokens, _ = self.run_stream('a', flags=['--flush-tokens', '3'], flush='bbb')
        self.assertEqual([event['token'] for event in tokens], ['a', 'b'])

    def test_eos_can_still_end_stream_during_another_tokens_cooldown(self):
        tokens, events = self.run_stream(['a', '<eos>'])
        self.assertEqual([event['token'] for event in tokens], ['a'])
        self.assertEqual([event['event'] for event in events], ['start', 'token', 'eos', 'done'])

    def test_negative_cooldown_rejected_before_loading_model(self):
        args = infer.build_argparser().parse_args([
            '--checkpoint', 'unused', '--video', 'unused', '--repeat-token-cooldown-frames', '-1'])
        with patch.object(infer, 'load_model') as load_model:
            with self.assertRaisesRegex(SystemExit, '--repeat-token-cooldown-frames must be >= 0'):
                infer.main(args)
        load_model.assert_not_called()


if __name__ == '__main__':
    unittest.main()
