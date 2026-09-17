#!/usr/bin/env python3
"""Compatibility entry point; new commands should use train_downvid.py."""

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("train_downvid.py")), run_name="__main__")
else:
    import train_downvid

    sys.modules[__name__] = train_downvid
