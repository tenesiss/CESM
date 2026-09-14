#!/usr/bin/env python3
"""Variant 2: original model + lambda_tcross * mean(ReLU(m - ||tcross_t||_2)^2)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train as trainer


# CLI flags override these defaults. The penalty takes an L2 norm over hidden
# features per query, including EOS and excluding padding.
DEFAULTS = {
    "output": "checkpoints/test_2_tcross_loss.pt",
    "metrics_csv": "auto",
    "variant_name": "test_2_tcross_loss",
    "lr": 3e-4,
    "batch_size": 2,
    "epochs": 50,
    "confidence_epochs": 5,
    "confidence_lr": 3e-4,
    "text_fusion": "residual",
    "lambda_tcross": 1.0,
    "tcross_margin": 1.0,  # m in the penalty above.
}


def build_argparser():
    parser = trainer.build_argparser()
    parser.description = __doc__
    parser.set_defaults(**DEFAULTS)
    return parser


if __name__ == "__main__":
    trainer.train(build_argparser().parse_args())
