#!/usr/bin/env python3
"""Prepare one video dataset, then train and evaluate selected variants on the same samples.

Use -N with --dataset talkvid/hdtf, or --manifest to reuse a prepared dataset. Training
options apply to all compatible selected variants; --variants selects which to run
(default: all five), and --variant-args provides per-variant
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
           "text_fusion", "variant_name", "metrics_csv", "validation_manifest", "pretrain_manifest"}
DOWNLOAD_ONLY = {"num_videos", "num_pretrain", "prepare_only", "login", "logout", "exclude_manifest"}
SPECIFIC = {"lambda_tcross": {2}, "tcross_margin": {2}, "w_t": {4},
            "vcross_weight": {5}, "window_phasing": {5},
            "lambda_mono": {1, 2, 3, 4}, "lambda_align": {1, 2, 3, 4},
            "flash_mono": {1, 2, 3, 4}}


def training_actions():
    return {a.dest: a for a in train.build_argparser()._actions if a.dest != "help"}


def copy_option(parser, action, *, default=argparse.SUPPRESS):
    """Copy the simple store/boolean/append options used by the existing CLIs."""
    options = dict(dest=action.dest, default=default, help=action.help)
    if isinstance(action, argparse.BooleanOptionalAction):
        options["action"] = argparse.BooleanOptionalAction
    elif isinstance(action, argparse._StoreTrueAction):
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
    flags = action.option_strings
    if isinstance(action, argparse.BooleanOptionalAction):
        flags = [flag for flag in flags if not flag.startswith("--no-")]
    parser.add_argument(*flags, **options)


def build_argparser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--num-videos", "-N", type=train_downvid.positive_int,
                        help="Download and align exactly N usable videos/clips once")
    source.add_argument("--manifest", type=Path,
                        help="Reuse these exact samples instead of downloading")
    validation = parser.add_mutually_exclusive_group()
    validation.add_argument("--N-valid", dest="num_valid", type=train_downvid.positive_int,
                            help="Download this many separate validation videos/clips; enables early stopping")
    validation.add_argument("--validation-manifest", type=Path,
                            help="Reuse a separate validation JSONL instead of downloading validation videos")
    pretraining = parser.add_mutually_exclusive_group()
    pretraining.add_argument("--N-pretrain", dest="num_pretrain", type=train_downvid.positive_int,
                             help="Download this many unannotated pretraining videos once for all selected variants")
    pretraining.add_argument("--pretrain-manifest", type=Path,
                             help="Reuse these pretraining videos; either pretraining option enables --pretrain-visual-encoder")
    parser.add_argument("--run-dir", type=Path,
                        help="New output directory (default: runs/variants_TIMESTAMP beside this script)")
    parser.add_argument("--variants", type=int, nargs="+", choices=range(1, len(VARIANTS) + 1),
                        default=list(range(1, len(VARIANTS) + 1)), metavar="N",
                        help="Variants to train and evaluate, e.g. --variants 1 3 5 (default: 1 2 3 4 5); run once each in numeric order")
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
    if isinstance(action, argparse.BooleanOptionalAction):
        return [] if value is None else [action.option_strings[0 if value else 1]]
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
    try:
        train.validate_early_stopping_args(args)
        train.validate_pretraining_args(args)
        train.accuracy_decode_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.epochs <= 0 or args.confidence_epochs < 0:
        parser.error("Each variant requires --epochs > 0 and --confidence-epochs >= 0")
    if args.pretrain_visual_encoder and not args.pretrain_manifest:
        parser.error("--pretrain-visual-encoder requires --N-pretrain or --pretrain-manifest")
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
    if args.vcross_weight is not None and not math.isfinite(args.vcross_weight):
        parser.error("--vcross-weight must be finite")
    if not math.isfinite(args.window_phasing) or not 0 <= args.window_phasing <= 1:
        parser.error("--window-phasing must be finite and in [0,1]")
    if args.fine_tune_pretrained_lm and not args.pretrained_lm:
        parser.error("--fine-tune-pretrained-lm requires shared --pretrained-lm")


def build_commands(parser, args, run_dir):
    """Build and validate every training command before downloading anything."""
    actions = training_actions()
    overrides = variant_overrides(parser, args.variant_args)
    selected = set(args.variants)
    for specification in args.variant_args:
        index = int(specification.partition(":")[0])
        if index not in selected:
            parser.error(f"--variant-args targets variant {index}, which is not selected by --variants")
    snapshot = run_dir / "samples.jsonl"
    pretraining = args.num_pretrain is not None or args.pretrain_manifest is not None
    stages = []
    prepared = args.manifest.expanduser().resolve() if args.manifest else run_dir / f"{args.dataset}.jsonl"
    def download_command(count, manifest, *, video_only=False):
        command = [sys.executable, str(ROOT / "train_downvid.py"), "--prepare-only",
                   f"--N-pretrain={count}" if video_only else f"--num-videos={count}",
                   f"--pretrain-manifest={manifest}" if video_only else f"--manifest={manifest}"]
        for action in train_downvid.build_argparser()._actions:
            if action.dest in MANAGED | DOWNLOAD_ONLY:
                continue
            if action.dest not in actions or (not video_only and action.dest in ("pretrained_lm", "pretrained_lm_local_files_only")):
                command.extend(option_tokens(action, getattr(args, action.dest, None)))
        return command
    if args.num_videos is not None:
        command = download_command(args.num_videos, prepared)
        if args.validation_manifest is not None:
            command.append(f"--exclude-manifest={run_dir / 'validation_samples.jsonl'}")
        stages.append({"name": "download", "command": command})
    if args.num_valid is not None:
        validation_prepared = run_dir / f"validation_{args.dataset}.jsonl"
        command = download_command(args.num_valid, validation_prepared)
        excluded = run_dir / "validation_exclusions.jsonl" if args.pretrain_manifest else snapshot
        command.append(f"--exclude-manifest={excluded}")
        stages.append({"name": "download_validation", "command": command})
    if args.num_pretrain is not None:
        command = download_command(args.num_pretrain, run_dir / f"pretrain_{args.dataset}.jsonl", video_only=True)
        if args.num_valid is not None or args.validation_manifest is not None:
            command.append(f"--exclude-manifest={run_dir / 'validation_samples.jsonl'}")
        stages.append({"name": "download_pretrain", "command": command})
    checkpoints = []
    for index, module in enumerate(VARIANTS, 1):
        if index not in selected:
            continue
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
        if args.num_valid is not None or args.validation_manifest is not None:
            flags.append(f"--validation-manifest={run_dir / 'validation_samples.jsonl'}")
        if pretraining:
            flags.extend(["--pretrain-visual-encoder", f"--pretrain-manifest={run_dir / 'pretrain_samples.jsonl'}"])
        parsed = module.build_argparser().parse_args(flags)
        validate_training_args(parser, parsed)
        training_stage = {"name": f"train_{index}", "command": [sys.executable, module.__file__, *flags]}
        if parsed.plot_learning_curves and any(
            getattr(parsed, "plot_" + curve) and (not curve.startswith("validation_") or parsed.validation_manifest)
            for curve in ("training_loss", "validation_loss", "training_accuracy", "validation_accuracy")
        ):
            training_stage.update(learning_plot=str(checkpoint.with_suffix(".learning.png")),
                                  learning_csv=str(checkpoint.with_suffix(".learning.csv")))
        stages.append(training_stage)
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
        if args.num_valid is not None or args.validation_manifest is not None:
            validation_csv = checkpoint.with_suffix(".validation.metrics.csv")
            validation_command = command.copy()
            validation_command[validation_command.index("--manifest") + 1] = str(run_dir / "validation_samples.jsonl")
            validation_command[validation_command.index("--output") + 1] = str(validation_csv)
            stages.append({"name": f"evaluate_validation_{index}", "command": validation_command,
                           "csv": str(validation_csv), "split": "validation"})
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


def snapshot_samples(source, destination, expected_count=None, *, video_only=False):
    rows = train.load_manifest(str(source), video_only=video_only)
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
              "variants": sorted(set(args.variants)),
              "checkpoints": list(map(str, checkpoints)), "csv": str(run_dir / "all_variants.csv"),
              "stages": [dict(stage, status="pending") for stage in stages]}
    validation_prepared = None
    if args.num_valid is not None or args.validation_manifest is not None:
        validation_prepared = (args.validation_manifest.expanduser().resolve() if args.validation_manifest
                               else run_dir / f"validation_{args.dataset}.jsonl")
        report.update(validation_source=str(validation_prepared), validation_samples=None,
                      validation_csv=str(run_dir / "all_variants.validation.csv"))
    pretrain_prepared = None
    if args.num_pretrain is not None or args.pretrain_manifest is not None:
        pretrain_prepared = (args.pretrain_manifest.expanduser().resolve() if args.pretrain_manifest
                             else run_dir / f"pretrain_{args.dataset}.jsonl")
        report.update(pretrain_source=str(pretrain_prepared), pretrain_samples=None)
    train_downvid.write_json(report_path, report)
    print(f"Run directory: {run_dir}", flush=True)
    try:
        completed_csvs = []
        completed_validation_csvs = []
        if args.manifest:
            report["samples"] = snapshot_samples(prepared, snapshot)
        if args.validation_manifest:
            report["validation_samples"] = snapshot_samples(validation_prepared, run_dir / "validation_samples.jsonl")
        if args.pretrain_manifest:
            report["pretrain_samples"] = snapshot_samples(pretrain_prepared, run_dir / "pretrain_samples.jsonl",
                                                          video_only=True)
        def verify_splits():
            for split in ("samples", "validation_samples", "pretrain_samples"):
                if report.get(split) is not None:
                    verify_snapshot(report[split])
            if report.get("validation_samples") is not None:
                validation_rows = train.load_manifest(report["validation_samples"]["manifest"])
                for split in ("samples", "pretrain_samples"):
                    if report.get(split) is not None:
                        train.validate_validation_split(
                            train.load_manifest(report[split]["manifest"], video_only=True), validation_rows,
                        )
        verify_splits()
        for stage in report["stages"]:
            verify_splits()
            stage["status"] = "running"
            stage["log"] = str(run_dir / "logs" / f"{stage['name']}.log")
            train_downvid.write_json(report_path, report)
            if stage["name"] == "download_validation" and args.pretrain_manifest:
                # Validation excludes both supervised videos and reused
                # unlabeled videos, including their original source uploads.
                excluded = []
                for split in ("samples", "pretrain_samples"):
                    excluded.extend(train.load_manifest(report[split]["manifest"], video_only=True))
                (run_dir / "validation_exclusions.jsonl").write_text(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in excluded), encoding="utf-8",
                )
            print(f"\n{stage['name']}: {shlex.join(stage['command'])}", flush=True)
            run_stage(stage["command"], Path(stage["log"]))
            if stage["name"] == "download":
                report["samples"] = snapshot_samples(prepared, snapshot, args.num_videos)
            if stage["name"] == "download_validation":
                report["validation_samples"] = snapshot_samples(validation_prepared,
                                                                run_dir / "validation_samples.jsonl", args.num_valid)
            if stage["name"] == "download_pretrain":
                report["pretrain_samples"] = snapshot_samples(pretrain_prepared, run_dir / "pretrain_samples.jsonl",
                                                              args.num_pretrain, video_only=True)
            verify_splits()
            if "csv" in stage:
                validation_stage = stage.get("split") == "validation"
                completed = completed_validation_csvs if validation_stage else completed_csvs
                output_csv = report["validation_csv"] if validation_stage else report["csv"]
                combine_metrics_csv([*completed, stage["csv"]], output_csv)
                completed.append(stage["csv"])
                print(f"Updated combined CSV: {output_csv} ({len(completed)}/{len(checkpoints)} variants)", flush=True)
            stage["status"] = "complete"
            train_downvid.write_json(report_path, report)
        verify_splits()
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
    print(f"\nCompleted {len(checkpoints)} selected variant(s) on {report['samples']['count']} samples.", flush=True)
    print(f"Combined CSV: {report['csv']}", flush=True)
    if "validation_csv" in report:
        print(f"Validation CSV: {report['validation_csv']}", flush=True)
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
