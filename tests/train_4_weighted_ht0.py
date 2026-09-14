#!/usr/bin/env python3
"""Variant 4: ht = text_ln(w_t * ht0 + tcross), with a fixed scalar w_t."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train as trainer


# CLI flags override these defaults.
# w_t is a checkpointed buffer, never an optimizer parameter.
DEFAULTS = {
    "output": "checkpoints/test_4_weighted_ht0.pt",
    "metrics_csv": "auto",
    "variant_name": "test_4_weighted_ht0",
    "lr": 3e-4,
    "batch_size": 2,
    "epochs": 50,
    "confidence_epochs": 5,
    "confidence_lr": 3e-4,
    "text_fusion": "weighted",
    "lambda_tcross": 0.0,
    "w_t": 1.0,  # 1.0 reproduces the original fusion.
}


def build_argparser():
    parser = trainer.build_argparser()
    parser.description = __doc__
    parser.set_defaults(**DEFAULTS)
    return parser


if __name__ == "__main__":
    trainer.train(build_argparser().parse_args())
