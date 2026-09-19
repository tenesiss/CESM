#!/usr/bin/env python3
"""Variant 5: ht = text_ln(hv0 + vcross_weight * vcross), with per-frame token logits."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train as trainer


# CLI flags override these defaults. Cross entropy, confidence targets and CSV
# metrics use real token windows; projection penalties also use unlabeled frames
# in samples with supervision, and the full valid teacher text prefix.
DEFAULTS = {
    "output": "checkpoints/test_5_frame_tokens.pt",
    "metrics_csv": "auto",
    "variant_name": "test_5_frame_tokens",
    "lr": 3e-4,
    "batch_size": 2,
    "epochs": 50,
    "confidence_epochs": 5,
    "confidence_lr": 3e-4,
    "text_fusion": "frame",
    "lambda_tcross": 0.0,
    "lambda_mono": 0.0,
    "lambda_align": 0.0,
}


def build_argparser():
    parser = trainer.build_argparser()
    parser.description = __doc__
    parser.set_defaults(**DEFAULTS)
    return parser


if __name__ == "__main__":
    trainer.train(build_argparser().parse_args())
