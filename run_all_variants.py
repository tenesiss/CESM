#!/usr/bin/env python3
"""Prepare one video dataset, then train and evaluate each variant on the same samples.

Use -N with --dataset talkvid/hdtf, or --manifest to reuse a prepared dataset. Training
options apply to all compatible variants; --variant-args provides per-variant
overrides. Each stage runs in its own process to release model/GPU memory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import train
import train_downvid
from tests import (
    train_1_as_is, train_2_tcross_loss, train_3_no_ht0,
    train_4_weighted_ht0, train_5_frame_tokens,
)


ROOT = Path(__file__).resolve().parent
VARIANTS = (train_1_as_is, train_2_tcross_loss, train_3_no_ht0,
            train_4_weighted_ht0, train_5_frame_tokens)
MANAGED = {"help", "manifest", "output", "resume", "confidence_only",
           "text_fusion", "variant_name", "metrics_csv"}
DOWNLOAD_ONLY = {"num_videos", "prepare_only", "login", "logout"}
SPECIFIC = {"lambda_tcross": {2}, "tcross_margin": {2}, "w_t": {4},
            "lambda_mono": {1, 2, 3, 4}, "lambda_align": {1, 2, 3, 4},
            "flash_mono": {1, 2, 3, 4}}


def training_actions():
    return {a.dest: a for a in train.build_argparser()._actions if a.dest != "help"}


def copy_option(parser, action, *, default=argparse.SUPPRESS):
    """Copy the simple store/boolean/append options used by the existing CLIs."""
    options = dict(dest=action.dest, default=default, help=action.help)
    if isinstance(action, argparse._StoreTrueAction):
        options["action"] = "store_true"
    elif isinstance(action, argparse._StoreFalseAction):
        options["action"] = "store_false"
    else:
        if isinstance(action, argparse._AppendAction):
            options["action"] = "append"
        for key in ("type", "choices", "metavar", "nargs"):
            value = getattr(action, key)
            if value is not None:
                options[key] = value
    parser.add_argument(*action.option_strings, **options)


def build_argparser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--num-videos", "-N", type=train_downvid.positive_int,
                        help="Download and align exactly N usable videos/clips once")
    source.add_argument("--manifest", type=Path,
                        help="Reuse these exact samples instead of downloading")
    parser.add_argument("--run-dir", type=Path,
                        help="New output directory (default: runs/variants_TIMESTAMP beside this script)")
    parser.add_argument("--variant-args", action="append", default=[], metavar="N:FLAGS",
                        help="Per-variant overrides, e.g. '4:--lr 1e-4 --batch-size 4'; repeatable")
    parser.add_argument("--eval-batch-size", type=train_downvid.positive_int, default=None,
                        help="Batch size for each evaluation (default: shared --batch-size or 2)")
    parser.add_argument("--eval-device", default=None,
                        help="Device for each evaluation (default: shared --device or auto)")
    shared = parser.add_argument_group("Shared training options (individual script defaults otherwise)")
    download = parser.add_argument_group("Dataset download and alignment options")
    cookies = download.add_mutually_exclusive_group()
    training = training_actions()
    for action in train_downvid.build_argparser()._actions:
        if action.dest in MANAGED | DOWNLOAD_ONLY:
            continue
        if action.dest in training:
            copy_option(shared, action)
        else:
            group = cookies if action.dest in ("cookies", "cookies_from_browser") else download
            copy_option(group, action, default=action.default)
    return parser


def option_tokens(action, value):
    if isinstance(action, argparse._StoreTrueAction):
        return [action.option_strings[0]] if value else []
    if isinstance(action, argparse._StoreFalseAction):
        return [] if value else [action.option_strings[0]]
    if value is None:
        return []
    values = value if isinstance(action, argparse._AppendAction) else [value]
    return [f"{action.option_strings[0]}={v}" for v in values]


def variant_overrides(parser, specifications):
    actions = training_actions()
    # Tokenization is shared with the downloader; metric semantics are shared
    # with the evaluator. Neither should diverge in one child process.
    protected = MANAGED | {"pretrained_lm", "metrics_include_eos"}
    option_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for dest, action in actions.items():
        if dest not in protected:
            copy_option(option_parser, action)
    result = {i: {} for i in range(1, 6)}
    for specification in specifications:
        number, separator, flags = specification.partition(":")
        if not separator or number not in {"1", "2", "3", "4", "5"}:
            parser.error("--variant-args must be N:FLAGS, where N is 1 through 5")
        try:
            values = vars(option_parser.parse_args(shlex.split(flags)))
        except ValueError as exc:
            parser.error(str(exc))
        index = int(number)
        for name in values:
            if name in SPECIFIC and index not in SPECIFIC[name]:
                parser.error(f"--{name.replace('_', '-')} does not apply to variant {index}")
        result[index].update(values)
    return result


def validate_training_args(parser, args):
    if args.epochs <= 0 or args.confidence_epochs < 0:
        parser.error("Each variant requires --epochs > 0 and --confidence-epochs >= 0")
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("--batch-size must be > 0 and --workers must be >= 0")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be > 0")
    for key in ("lr", "confidence_lr", "grad_clip", "confidence_beta",
                "confidence_token_temperature", "confidence_temperature"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be finite and > 0")
    for key in ("weight_decay", "lambda_tcross", "tcross_margin", "confidence_future_weight_decay"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            parser.error(f"--{key.replace('_', '-')} must be finite and >= 0")
    if not math.isfinite(args.w_t) or not 0 <= args.confidence_lambda <= 1:
        parser.error("--w-t must be finite and --confidence-lambda must be in [0,1]")
    if args.fine_tune_pretrained_lm and not args.pretrained_lm:
        parser.error("--fine-tune-pretrained-lm requires shared --pretrained-lm")


def build_commands(parser, args, run_dir):
    """Build and validate every training command before downloading anything."""
    actions = training_actions()
    overrides = variant_overrides(parser, args.variant_args)
    snapshot = run_dir / "samples.jsonl"
    stages = []
    prepared = args.manifest.expanduser().resolve() if args.manifest else run_dir / f"{args.dataset}.jsonl"
    if args.num_videos is not None:
        command = [sys.executable, str(ROOT / "train_downvid.py"), "--prepare-only",
                   f"--num-videos={args.num_videos}", f"--manifest={prepared}"]
        for action in train_downvid.build_argparser()._actions:
            if action.dest in MANAGED | DOWNLOAD_ONLY:
                continue
            if action.dest not in actions or action.dest in ("pretrained_lm", "pretrained_lm_local_files_only"):
                command.extend(option_tokens(action, getattr(args, action.dest, None)))
        stages.append({"name": "download", "command": command})
    checkpoints = []
    for index, module in enumerate(VARIANTS, 1):
        checkpoint = run_dir / "checkpoints" / Path(module.DEFAULTS["output"]).name
        checkpoints.append(checkpoint)
        values = {dest: getattr(args, dest) for dest in actions
                  if dest not in MANAGED and hasattr(args, dest)
                  and (dest not in SPECIFIC or index in SPECIFIC[dest])}
        values.update(overrides[index])
        flags = [token for dest, value in values.items() for token in option_tokens(actions[dest], value)]
        # Evaluate the saved checkpoint in a separate process immediately after training.
        flags += [f"--manifest={snapshot}", f"--output={checkpoint}", "--metrics-csv=",
                  f"--variant-name={module.DEFAULTS['variant_name']}"]
        parsed = module.build_argparser().parse_args(flags)
        validate_training_args(parser, parsed)
        stages.append({"name": f"train_{index}", "command": [sys.executable, module.__file__, *flags]})
        metrics_csv = checkpoint.with_suffix(".metrics.csv")
        command = [sys.executable, str(ROOT / "tests" / "evaluate_checkpoints.py"),
                   "--manifest", str(snapshot), "--checkpoints", str(checkpoint),
                   "--output", str(metrics_csv),
                   "--batch-size", str(args.eval_batch_size or getattr(args, "batch_size", 2)),
                   "--device", args.eval_device or getattr(args, "device", "auto"),
                   "--workers", str(getattr(args, "workers", 0))]
        if getattr(args, "metrics_include_eos", False):
            command.append("--include-eos")
        # Each checkpoint supplies its own preprocessing and max_frames, including
        # per-variant overrides. The same sample manifest is always used.
        stages.append({"name": f"evaluate_{index}", "command": command, "csv": str(metrics_csv)})
    return prepared, snapshot, checkpoints, stages


def combine_metrics_csv(paths, output):
    """Publish completed variants together without risking the previous combined CSV."""
    rows = []
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    output = Path(output)
    temporary = output.with_suffix(".csv.tmp")
    try:
        train.write_metrics_csv(temporary, rows)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def snapshot_samples(source, destination, expected_count=None):
    rows = train.load_manifest(str(source))
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"Prepared {len(rows)} samples; expected exactly {expected_count}. Training was not started.")
    for row in rows:
        row["video"] = str(Path(row["video"]).resolve())
        if not Path(row["video"]).is_file():
            raise FileNotFoundError(f"Sample video does not exist: {row['video']}")
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    destination.write_bytes(payload)
    return {"manifest": str(destination), "sha256": hashlib.sha256(payload).hexdigest(), "count": len(rows)}


def verify_snapshot(snapshot):
    actual = hashlib.sha256(Path(snapshot["manifest"]).read_bytes()).hexdigest()
    if actual != snapshot["sha256"]:
        raise RuntimeError("The saved sample manifest changed during this run; refusing to mix datasets")


def run_stage(command, log_path):
    """Stream progress to the console and retain a separate log for each stage."""
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", bufsize=1,
                              env=dict(os.environ, PYTHONUNBUFFERED="1")) as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
                if code:
                    raise subprocess.CalledProcessError(code, command)
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                raise


def main(argv=None):
    parser = build_argparser()
    args = parser.parse_args(argv)
    run_dir = (args.run_dir or ROOT / "runs" / datetime.now().strftime("variants_%Y%m%d_%H%M%S_%f")).expanduser().resolve()
    prepared, snapshot, checkpoints, stages = build_commands(parser, args, run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        parser.error(f"Run directory is not empty: {run_dir}. Choose a new --run-dir to keep runs separate.")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir()
    report_path = run_dir / "run.json"
    report = {"status": "running", "source": str(prepared), "samples": None,
              "checkpoints": list(map(str, checkpoints)), "csv": str(run_dir / "all_variants.csv"),
              "stages": [dict(stage, status="pending") for stage in stages]}
    train_downvid.write_json(report_path, report)
    print(f"Run directory: {run_dir}", flush=True)
    try:
        completed_csvs = []
        if args.manifest:
            report["samples"] = snapshot_samples(prepared, snapshot)
        for stage in report["stages"]:
            if report["samples"] is not None:
                verify_snapshot(report["samples"])
            stage["status"] = "running"
            stage["log"] = str(run_dir / "logs" / f"{stage['name']}.log")
            train_downvid.write_json(report_path, report)
            print(f"\n{stage['name']}: {shlex.join(stage['command'])}", flush=True)
            run_stage(stage["command"], Path(stage["log"]))
            if stage["name"] == "download":
                report["samples"] = snapshot_samples(prepared, snapshot, args.num_videos)
            if "csv" in stage:
                combine_metrics_csv([*completed_csvs, stage["csv"]], report["csv"])
                completed_csvs.append(stage["csv"])
                print(f"Updated combined CSV: {report['csv']} ({len(completed_csvs)}/5 variants)", flush=True)
            stage["status"] = "complete"
            train_downvid.write_json(report_path, report)
        verify_snapshot(report["samples"])
        report["status"] = "complete"
        train_downvid.write_json(report_path, report)
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        for stage in report["stages"]:
            if stage["status"] == "running":
                stage["status"] = report["status"]
        report["error"] = str(exc)
        train_downvid.write_json(report_path, report)
        raise
    print(f"\nCompleted all five variants on {report['samples']['count']} samples.", flush=True)
    print(f"Combined CSV: {report['csv']}", flush=True)
    return report


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; completed stages and logs remain in the run directory.", file=sys.stderr)
        sys.exit(130)
    except subprocess.CalledProcessError as exc:
        print(f"Stage failed (exit {exc.returncode}); see its log in the run directory.", file=sys.stderr)
        sys.exit(exc.returncode if exc.returncode > 0 else 1)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
