#!/usr/bin/env python3
"""Write one combined per-sample CSV for existing variant checkpoints."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import infer
import train


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="results/variant_metrics.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Override each checkpoint's saved section size")
    parser.add_argument("--include-eos", action="store_true")
    parser.add_argument("--no-face-detector", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                        help="Override CUDA AMP for evaluation (default: each checkpoint's training setting)")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.workers < 0 or (args.max_frames is not None and args.max_frames <= 0):
        parser.error("batch size and max frames must be positive; workers must be nonnegative")
    if Path(args.output).resolve() in {Path(p).resolve() for p in [args.manifest, *args.checkpoints]}:
        parser.error("output must differ from the checkpoint and manifest paths")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    samples = train.load_manifest(args.manifest)
    results = []
    for checkpoint in args.checkpoints:
        model, tokenizer, preprocess, saved = infer.load_model(checkpoint, device)
        saved.pop("model_state", None)  # Model already owns a copy of these weights.
        amp = args.amp if args.amp is not None else saved.get("training_args", {}).get("amp", False)
        max_frames = args.max_frames if args.max_frames is not None else saved.get("sectioning", {}).get("max_frames")
        dataset = train.VideoTextWindowDataset(
            samples, tokenizer, model.cfg.mouth_size,
            use_face_detector=preprocess.get("use_face_detector", True) and not args.no_face_detector,
            max_frames=max_frames,
            video_dtype=torch.get_autocast_dtype(device.type) if amp and device.type == "cuda" else torch.float32,
        )
        results.extend(train.evaluate_samples(
            model, dataset, device=device, batch_size=args.batch_size, workers=args.workers,
            variant=saved.get("training_args", {}).get("variant_name") or Path(checkpoint).stem,
            checkpoint=checkpoint, include_eos=args.include_eos, amp=amp,
        ))
        print(f"evaluated {checkpoint}: {len(samples)} samples")
        del model, saved, dataset, tokenizer
    train.write_metrics_csv(args.output, results)
    print(f"saved {args.output}: {len(results)} rows")


if __name__ == "__main__":
    main()
