#!/usr/bin/env python3
"""Variant 1: original ht = text_ln(ht0 + tcross), with the original losses."""

import sys
from pathlib import Path

# Permit `python /path/to/CESM/tests/train_1_as_is.py` from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train as trainer


# CLI flags override these defaults; keys use train.py's argparse destination names.
DEFAULTS = {
    "output": "checkpoints/test_1_as_is.pt",
    "metrics_csv": "auto",
    "variant_name": "test_1_as_is",
    "lr": 3e-4,
    "batch_size": 2,
    "epochs": 50,
    "confidence_epochs": 5,
    "confidence_lr": 3e-4,
    "text_fusion": "residual",
    "lambda_tcross": 0.0,
}


def build_argparser():
    parser = trainer.build_argparser()
    parser.description = __doc__
    parser.set_defaults(**DEFAULTS)
    return parser


if __name__ == "__main__":
    trainer.train(build_argparser().parse_args())
