"""Per-epoch CSV history and headless learning-curve rendering."""

import argparse
import csv
from pathlib import Path

from streaming import add_decode_arguments, validate_decode_arguments


CURVES = ("training_loss", "validation_loss", "training_accuracy", "validation_accuracy")
COLORS = ("#2563eb", "#ea580c", "#15803d", "#be123c")
HISTORY_COLUMNS = ("step", "stage", "epoch", "confidence_trained", *CURVES)


def add_learning_curve_arguments(parser):
    group = parser.add_argument_group("Learning curves and streaming accuracy")
    group.add_argument("--plot-learning-curves", action=argparse.BooleanOptionalAction, default=True,
                       help="Save CHECKPOINT_STEM.learning.png and .learning.csv after every epoch (default: on)")
    for curve in CURVES:
        group.add_argument("--plot-" + curve.replace("_", "-"),
                           action=argparse.BooleanOptionalAction, default=True,
                           help=f"Record and plot {curve.replace('_', ' ')} (default: on)")
    add_decode_arguments(group, prefix="accuracy-")


def accuracy_decode_args(args):
    options = argparse.Namespace(**{key.removeprefix("accuracy_"): value
                                   for key, value in vars(args).items()
                                   if key.startswith("accuracy_")})
    if options.token_temperature is None:
        options.token_temperature = args.confidence_token_temperature
    if options.conf_temperature is None:
        options.conf_temperature = args.confidence_temperature
    try:
        validate_decode_arguments(options)
    except ValueError as exc:
        raise ValueError(str(exc).replace("--", "--accuracy-")) from exc
    return options


def token_edit_distance(reference, prediction):
    """Levenshtein distance, using memory linear in the shorter sequence."""
    if len(reference) < len(prediction):
        reference, prediction = prediction, reference
    previous = list(range(len(prediction) + 1))
    for i, expected in enumerate(reference, 1):
        current = [i]
        for j, emitted in enumerate(prediction, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (expected != emitted)))
        previous = current
    return previous[-1]


class LearningCurveLogger:
    def __init__(self, args):
        self.selected = [curve for curve in CURVES
                         if args.plot_learning_curves and getattr(args, "plot_" + curve)
                         and (not curve.startswith("validation_") or args.validation_manifest)]
        self.csv_path = Path(args.output).with_suffix(".learning.csv")
        self.plot_path = Path(args.output).with_suffix(".learning.png")
        self.title = args.variant_name or Path(args.output).stem
        self.history = []
        if self.selected:
            # Check this dependency before training; no GUI backend is needed.
            try:
                from matplotlib.backends.backend_agg import FigureCanvasAgg
                from matplotlib.figure import Figure
            except ImportError as exc:
                raise RuntimeError("Learning curves require matplotlib; install requirements.txt "
                                   "or pass --no-plot-learning-curves") from exc
            self.figure_class = Figure
            self.canvas_class = FigureCanvasAgg

    def record(self, stage, epoch, confidence_trained, **values):
        record = dict(step=len(self.history) + 1, stage=stage, epoch=epoch,
                      confidence_trained=confidence_trained)
        record.update({curve: values.get(curve) if curve in self.selected else None for curve in CURVES})
        self.history.append(record)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=HISTORY_COLUMNS)
            writer.writeheader()
            writer.writerows(self.history)
        self.plot()

    def plot(self):
        # Separate stages because token and confidence loss have different units.
        stages = list(dict.fromkeys(row["stage"] for row in self.history))
        kinds = [kind for kind in ("loss", "accuracy")
                 if any(curve.endswith(kind) for curve in self.selected)]
        fig = self.figure_class(figsize=(7 * len(stages), 3.5 * len(kinds)), layout="constrained")
        self.canvas_class(fig)
        axes = fig.subplots(len(kinds), len(stages), squeeze=False)
        for col, stage in enumerate(stages):
            records = [row for row in self.history if row["stage"] == stage]
            for row_index, kind in enumerate(kinds):
                axis = axes[row_index, col]
                for curve, color in zip(CURVES, COLORS):
                    if curve not in self.selected or not curve.endswith(kind):
                        continue
                    points = [(row["epoch"], row[curve]) for row in records if row[curve] is not None]
                    if points:
                        x, y = zip(*points)
                        training = curve.startswith("training_")
                        axis.plot(x, [v * 100 if kind == "accuracy" else v for v in y],
                                  label=curve.replace("_", " ").capitalize(), color=color,
                                  linestyle="-" if training else "--", marker="o" if training else "x")
                axis.set(title=f"{stage.capitalize()} stage", xlabel="Epoch",
                         ylabel="Token edit accuracy (%)" if kind == "accuracy" else "Objective loss")
                axis.xaxis.get_major_locator().set_params(integer=True)
                if kind == "accuracy":
                    axis.set_ylim(-3, 103)
                    axis.set_yticks(range(0, 101, 20))
                axis.grid(True, alpha=0.2)
                if axis.lines:
                    axis.legend()
        fig.suptitle(self.title)
        fig.savefig(self.plot_path, dpi=150)
        fig.clear()
