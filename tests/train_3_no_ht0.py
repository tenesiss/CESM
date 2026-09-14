#!/usr/bin/env python3
"""Variant 3: ht = text_ln(tcross), with the original losses."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train as trainer


# CLI flags override these defaults.
# ht0 still supplies text attention queries; only its residual term is removed.
DEFAULTS = {
    "output": "checkpoints/test_3_no_ht0.pt",
    "metrics_csv": "auto",
    "variant_name": "test_3_no_ht0",
    "lr": 3e-4,
    "batch_size": 2,
    "epochs": 50,
    "confidence_epochs": 5,
    "confidence_lr": 3e-4,
    "text_fusion": "cross_only",
    "lambda_tcross": 0.0,
}


def build_argparser():
    parser = trainer.build_argparser()
    parser.description = __doc__
    parser.set_defaults(**DEFAULTS)
    return parser


if __name__ == "__main__":
    trainer.train(build_argparser().parse_args())
