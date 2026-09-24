# Architecture variants

Each script has its own editable `DEFAULTS` dictionary and accepts every
`train.py` CLI option. CLI values override the dictionary. The scripts share the
model/training implementation in `../train.py`; keep them inside this repository.

| Script | Text fusion | Additional token-stage loss |
| --- | --- | --- |
| `train_1_as_is.py` | `text_ln(ht0 + tcross)` | None |
| `train_2_tcross_loss.py` | `text_ln(ht0 + tcross)` | `lambda_tcross * mean(ReLU(m - norm(tcross_t, 2))**2)` |
| `train_3_no_ht0.py` | `text_ln(tcross)` | None |
| `train_4_weighted_ht0.py` | `text_ln(w_t * ht0 + tcross)` | None |
| `train_5_frame_tokens.py` | `text_ln(hv0 + vcross_weight * vcross)` | Frame cross entropy replaces token cross entropy |

The original token loss, projection losses, attention losses, text encoder and
confidence stage keep their existing defaults. Variant 3 removes only the `ht0`
residual; `ht0` still forms the queries in text-to-video attention.

For variant 2, `tcross` is the cross-attention output **before** text layer norm.
The L2 norm is computed across its hidden features for each token, then the
squared hinge penalties are averaged over all valid token queries in the batch,
including a true EOS target. Padding and unsupervised text history do not count.
The norm/penalty accumulates in FP32 under AMP, with gradients into cross-attention.
`--tcross-margin M` sets the nonnegative threshold `m` (default `1.0`);
`--lambda-tcross` sets the loss coefficient (default `1.0` for variant 2, `0.0`
for the others). The raw added loss is logged as `tcross`.

For variant 4, `--w-t` sets a **fixed scalar**, stored in the checkpoint and never
updated by the optimizer. The default `1.0` reproduces the baseline fusion;
choose another value for a distinct experiment. `0.0` has the same fusion as
variant 3. The current value is logged as `w_t`.

For variant 5, `vcross = video_cross(hv0, ht0)` uses frames as queries and
causally available text as keys/values. `--vcross-weight` sets its fixed multiplier
in token fusion (default `1.0`; `0.0` gives video-only token predictions).
The text FFN and token head then produce
`[B, F, V]` logits. There is no `text_cross` module or `tcross` computation.
Each frame inside a supplied token window is labeled with that token. Unlabeled
frames and padding are excluded from both cross entropy and metrics; no EOS label
overwrites the last spoken token. Windows crossing a `--max-frames` boundary
supervise frames on both sides. The current token is hidden from attention until
the frame after its window ends, preserving the existing causal teacher forcing.

Variant 5 also accepts `--window-phasing FRACTION` (default `0`, range `[0,1]`).
For the first `floor(FRACTION * original_window_length)` frames of each window
after the first, the frame loss is `0.5 * CE(current) + 0.5 * CE(previous)`.
Equivalently, the last phased zero-based offset is
`floor(FRACTION * original_window_length) - 1`. Each frame still counts once
in the batch mean. Phasing never restarts at a section boundary and never labels
gaps or padding. Current-token CE sees text through the previous token;
previous-token CE excludes that token and all later text. Repeated token IDs
still use these distinct contexts. The extra path shares encoder outputs and
Q/K/V projections, and checkpoints attention/head/CE in chunks of at most 128
phased frames. Metrics keep scoring the current token; confidence windows and
the current path's causal teacher timing stay the same.
The checkpoint records the fraction; pass it again on resume to retain it.

Projection losses retain their defaults. `--lambda-tcross`, `--lambda-mono`, and
`--lambda-align` must remain zero in variant 5 because there is no text-to-video
attention. `--flash-mono` has no effect on this variant. Its confidence stage
also freezes `video_cross`, which now participates in token prediction, and
uses frame distributions within the labeled windows for its targets.

All variants support `--no-vproj`, which uses the video encoder features as
`hv0` directly, sets fusion width to `--d-video` instead of `--d-fusion`, and
projects text to that width. Video projection penalties are skipped in this
mode. The checkpoint saves this architecture and the frame `vcross` weight;
inference and resume restore them. The confidence branch retains its own text
attention, even when the frame token-fusion weight is zero.

## Run

Install the repository's `requirements.txt` in your Python environment. From the
repository root, use your existing JSONL manifest (replace `data/train.jsonl`):

```bash
python3 tests/train_1_as_is.py --manifest data/train.jsonl --lr 3e-4 --batch-size 2
python3 tests/train_2_tcross_loss.py --manifest data/train.jsonl --lr 3e-4 --batch-size 2 --tcross-margin 1.0 --lambda-tcross 1.0
python3 tests/train_3_no_ht0.py --manifest data/train.jsonl --lr 3e-4 --batch-size 2
python3 tests/train_4_weighted_ht0.py --manifest data/train.jsonl --lr 3e-4 --batch-size 2 --w-t 0.5
python3 tests/train_5_frame_tokens.py --manifest data/train.jsonl --lr 3e-4 --batch-size 2
```

The fourth example explicitly chooses `0.5`; the script default remains `1.0`.
Each script writes a separate `checkpoints/test_N_*.pt` by default. Change
`--output` for additional parameter sweeps. Match the manifest, seed, batch size,
learning rate and epoch counts when comparing the architecture alone.

All existing options work, including `--pretrained-lm`, `--max-frames`, `--amp`,
`--flash-mono`, `--device`, and `--confidence-lr`. Use `--confidence-epochs 0` to
run only token training; otherwise the original five confidence epochs run after
the original fifty token epochs. Use `--help` on any script for the full CLI.
Scripts can also be invoked by absolute path from another directory; relative
manifest and output paths resolve from the current working directory.

The individual scripts also accept `--N-valid N` to hold out exactly N rows
from `--manifest`, using `--seed` and keeping videos/source uploads together.
This enables validation curves and early stopping. It is mutually exclusive
with `--validation-manifest`. The runner below instead downloads N additional
validation clips, preserving the number of training clips.

## All-in-one download, train and evaluate

From the repository root, run `run_all_variants.py` to download and align TalkVid, HDTF, or Shofo
**once**, then train and evaluate the selected variants (1–5 by default) sequentially on **the exact same
manifest samples**. Each variant's CSV is saved before the next variant starts:

```bash
python3 -m pip install -r requirements-downvid.txt
python3 run_all_variants.py \
  -N 20 --dataset-language English \
  --work-dir data/talkvid --run-dir runs/talkvid_comparison \
  --download-max-frames 256 --max-frames 128 \
  --epochs 50 --confidence-epochs 5 --batch-size 2 --lr 3e-4 \
  --tcross-margin 1.0 --lambda-tcross 1.0 --w-t 0.5
```

For HDTF, add `--dataset hdtf` and use `--work-dir data/hdtf`. For example:

```bash
python3 run_all_variants.py --dataset hdtf -N 100 \
  --download-max-frames 256 --max-frames 128 --language en \
  --run-dir runs/hdtf_comparison --epochs 50 --batch-size 2
```

HDTF also accepts `--hdtf-archive PATH_OR_URL` for a local or alternate ZIP.
The download stage calls `train_downvid.py` once and shares the resulting exact
sample manifest with all variants. HDTF transfers each selected compressed ZIP
member, then applies the frame cap locally and trims audio before alignment.

For [Shofo/shofo-talking-head-en](https://huggingface.co/datasets/Shofo/shofo-talking-head-en),
request/accept dataset access and authenticate with `hf auth login` or `HF_TOKEN`,
then use `--dataset shofo` (default cache: `data/shofo`). For example:

```bash
python3 run_all_variants.py --dataset shofo -N 100 --N-valid 20 \
  --download-max-frames 256 --max-frames 128 --language en \
  --run-dir runs/shofo_comparison --epochs 50 --batch-size 2
```

`--shofo-revision` selects a branch, tag, or commit. Each selected MP4 is fetched
in full and optionally trimmed locally; Hugging Face access failures stop the
run. See the [Shofo setup instructions](../README.md#shofo-hosted-english-talking-head-videos).

HDTF and Shofo default to four concurrent downloads. Set `--download-workers N`
to tune download concurrency. All sources also default to `--alignment-workers 4`
for parallel transcription and frame grouping. Use `--sequential-alignment` to
force one alignment worker while keeping parallel downloads, or add
`--download-workers 1` for fully serial preparation. The stages overlap while
retaining source order and exact usable sample counts. These options are forwarded
to all preparation stages; pretraining preparation skips alignment. Shofo uses
the Hugging Face Hub's Xet-capable downloader; HDTF keeps selective HTTP ZIP ranges
with independent readers. Alignment workers share a model configured for concurrent
transcriptions; higher alignment worker counts use more RAM/VRAM.

Install FFmpeg (`ffmpeg` and `ffprobe`); only TalkVid requires a supported
yt-dlp JavaScript runtime, as described in the repository README. All TalkVid download/alignment options
are available, including repeated `--dataset-language`, `--metadata`,
`--whisper-model`, `--whisper-device`, `--language`, `--cookies`,
`--cookies-from-browser`, `--js-runtimes`, and `--download-max-frames`.
The downloader reuses its existing clip/alignment cache and
replaces unusable candidates until it has N usable samples. An incomplete or
failed preparation stops the runner before any variant is trained.

To reuse an already prepared dataset without downloading or loading Whisper:

```bash
python3 run_all_variants.py \
  --manifest data/talkvid/data.jsonl --run-dir runs/existing_samples \
  --epochs 10 --batch-size 2 --max-frames 128 --w-t 0.5
```

To train and generate metrics for a subset, add `--variants` followed by one or
more variant numbers:

```bash
python3 run_all_variants.py --manifest data/talkvid/data.jsonl \
  --variants 1 3 5 --run-dir runs/selected_variants --epochs 10
```

Use `--variants 4` for a single variant. The default is all five. Selected variants
run once each in numeric order, even if numbers are repeated or supplied out of
order. Only selected variants get checkpoints, evaluation CSVs, and training/evaluation
logs; `all_variants.csv` combines only their results. `--variant-args` must target
a selected variant; otherwise the runner fails before downloading or training.

### Validation, early stopping, and checkpoint selection

`-N` remains the number of training clips. Add `--N-valid` to download a separate
validation set once, shared by every selected variant:

```bash
python3 run_all_variants.py --dataset hdtf -N 100 --N-valid 20 \
  --variants 1 4 5 --run-dir runs/hdtf_validation \
  --epochs 50 --confidence-epochs 10 --early-stopping-patience 5 \
  --early-stopping-min-delta 0.001
```

Validation preparation excludes training video paths and known source uploads,
so other TalkVid clips from a training upload are also excluded. Newly generated
manifests record `source_key` for this purpose. Legacy/custom manifests without
source information can only be checked by resolved video path; use separate
source videos when preparing them. Failed candidates do not cause the sets to
overlap. Both requested counts must be fully prepared before training starts.

To reuse existing datasets, pass `--manifest training.jsonl --validation-manifest
validation.jsonl`. You can also combine `--manifest` with `--N-valid`, or `-N`
with `--validation-manifest`. `--N-valid` and `--validation-manifest` are mutually
exclusive. Snapshots are saved as `samples.jsonl` and `validation_samples.jsonl`;
their hashes and counts are recorded in `run.json`. Overlapping splits are rejected.

After each epoch, the trainer evaluates the **same current weights** on both
datasets with dropout disabled and no gradient updates. The prediction stage uses
its total training objective, including enabled regularizers and window phasing;
the confidence stage uses its confidence objective. Each dataset's batch losses
are weighted by its number of supervised positions (tokens, frames, or confidence
timesteps). The two resulting dataset means have equal weight in checkpoint
selection, regardless of dataset sizes:

```text
checkpoint score = (training_loss + validation_loss) / 2
```

The checkpoint with the strictly lowest score is kept; ties keep the earlier
checkpoint. This is a combined criterion: the minimum training loss and minimum
validation loss may occur at different epochs. Early stopping separately watches
validation loss. `--early-stopping-patience` defaults to **5** epochs without an
improvement larger than `--early-stopping-min-delta` (default **0**). A patience
of **0** disables stopping while retaining best-checkpoint selection. These
settings also work in `--variant-args` and in the individual training scripts.

The best prediction weights are restored before confidence training. Confidence
training freezes that prediction path and selects its own best checkpoint with a
fresh patience counter. The best weights are restored even when the epoch limit
is reached. Loss evaluation adds a full training-set and validation-set pass per
epoch. Without a validation manifest, training runs its fixed epoch count.
Learning curves add end-of-epoch loss and streaming accuracy evaluation by default;
see [learning curves and decoder flags](../README.md#learning-curves-and-streaming-accuracy)
for all four curve toggles, inference settings, output paths, and score definitions.

Each final checkpoint records `validation_selection` with the best epoch, training
loss, validation loss, combined score, full epoch history, and stopping status for
each trained stage. Logs print the same losses and restored epoch numbers.
Final evaluation uses the restored model. Training metrics retain their existing
paths; validation metrics go to `CHECKPOINT_STEM.validation.metrics.csv` and
`RUN_DIR/all_variants.validation.csv`. These NLL/margin CSVs use the metric
definitions below, which differ from the total objective used for checkpoint selection.

Choose either `-N` or `--manifest`. `--work-dir` controls the reusable dataset
cache (default: `data/talkvid`, `data/hdtf`, or `data/shofo`); `--run-dir` holds this experiment's outputs and must be empty or new.
If omitted, a new timestamped directory under `runs/` is used. Paths passed on
the CLI resolve from the current directory, so invoking the runner by its
absolute path from elsewhere also works.

With all five variants selected, the runner creates:

```text
RUN_DIR/
  samples.jsonl                    # Fixed sample order, labels and absolute video paths
  <dataset>.jsonl                  # talkvid.jsonl, hdtf.jsonl, or shofo.jsonl in download mode
  checkpoints/test_1_as_is.pt
  checkpoints/test_2_tcross_loss.pt
  checkpoints/test_3_no_ht0.pt
  checkpoints/test_4_weighted_ht0.pt
  checkpoints/test_5_frame_tokens.pt
  checkpoints/test_1_as_is.metrics.csv  # One CSV beside each completed checkpoint
  ...
  all_variants.csv                 # Updated after each variant; five rows per sample when complete
  run.json                        # Selected variants, manifest SHA-256, stage status, commands and output paths
  logs/download.log               # Download mode only
  logs/train_1.log ... train_5.log
  logs/evaluate_1.log ... evaluate_5.log
```

`samples.jsonl` is copied once from the completed preparation (or input manifest),
with absolute video paths. Every training and evaluation command uses
that snapshot, and its SHA-256 is checked between stages. The video files remain
in their original/download locations. Keep those files for later evaluation.
Training processes exit before the next stage starts, releasing Whisper and
model memory between stages. After each variant finishes training, the runner
evaluates its saved checkpoint in a separate process and writes its
`CHECKPOINT_STEM.metrics.csv`. It then atomically updates `all_variants.csv`
with every completed variant's results before starting the next training stage.
The individual scripts' automatic CSV passes are disabled to avoid duplicate evaluation.

Unspecified settings retain each variant script's `DEFAULTS`. Shared flags such
as `--lr`, `--batch-size`, `--epochs`, `--pretrained-lm`, `--amp`, and `--seed`
apply to all variants. Experiment-specific options route as follows:

| Shared CLI option | Variants receiving it |
| --- | --- |
| `--tcross-margin`, `--lambda-tcross` | 2 only |
| `--w-t` | 4 only |
| `--vcross-weight`, `--window-phasing` | 5 only |
| `--no-vproj` | All variants |
| `--lambda-mono`, `--lambda-align`, `--flash-mono` | 1–4 only |

Use repeated `--variant-args 'N:FLAGS'` for independent hyperparameters:

```bash
python3 run_all_variants.py -N 20 --run-dir runs/custom_variants \
  --lr 3e-4 --batch-size 2 --max-frames 128 \
  --variant-args '2:--lr 1e-4 --tcross-margin 2.0' \
  --variant-args '4:--w-t 0.25 --batch-size 4' \
  --variant-args '5:--lr 5e-4 --max-frames 64'
```

Per-variant arguments override shared flags. Dataset paths, checkpoint paths,
fusion mode, model/tokenizer selection (`--pretrained-lm`), and metric EOS
settings are controlled at the runner level. The runner starts fresh training;
use each individual script's `--resume` to continue a saved checkpoint.

Evaluation uses each checkpoint's own preprocessing and section size, so
per-variant `--no-face-detector` or `--max-frames` settings are preserved.
Use `--eval-batch-size` or `--eval-device` to change evaluation resources;
otherwise evaluation uses the shared batch size/device (or `2`/`auto`).
`--metrics-include-eos` applies consistently to each variant's token metrics.

The combined CSV has the per-sample NLL and probability-margin columns described
below: token averages for variants 1–4, labeled-frame averages for variant 5.
These are measurements on the downloaded **training samples**. For N samples,
the completed CSV contains K × N data rows for K selected variants (5 × N by default),
even when videos are split into sections.
During the run, it contains N rows per successfully evaluated variant.
If a stage fails, later stages do not run; completed checkpoints, per-variant CSVs,
the combined CSV, logs and stage status remain available. Once all checkpoints exist, the standalone
evaluator below can repeat evaluation using `RUN_DIR/samples.jsonl`.

## Per-sample CSV after training

All five experiment scripts automatically evaluate the final model after both
training stages and write `CHECKPOINT_STEM.metrics.csv`, for example
`checkpoints/test_5_frame_tokens.metrics.csv`. Override the location with
`--metrics-csv results/my_run.csv`. Changing `--output` also changes the automatic
CSV path. The base `train.py` can opt in using `--metrics-csv auto`.

There is one row per **original manifest sample**, in manifest order, even when
training splits that video into sections. Separate rows with the same video path
remain separate samples. The CSV columns are:

```csv
variant,checkpoint,sample_index,video,text,unit,count,avg_nll,avg_margin
```

- `sample_index`: zero-based manifest sample index (blank lines do not count).
- `unit`: `token` for variants 1–4; `frame` for variant 5.
- `count`: number of transcript tokens or labeled frames evaluated.
- `avg_nll`: mean `-log(z_j)`, using natural logarithms (nats).
- `avg_margin`: mean `z_j - max(z_i for i != j)`; negative means a wrong class
  has higher probability, positive means the true class leads, and ties give zero.

Here `z = softmax(logits)` over the entire vocabulary at temperature 1, and `j`
is the true class. Evaluation uses the true preceding text, dropout disabled,
and no regularization terms. Automatic training CSVs describe the training split;
validation CSVs and the standalone evaluator describe their supplied splits.
These are teacher-forced scores, not free-running transcription metrics.
For variants 1–4, each transcript token
is evaluated at its supervised window end; BOS, padding, and EOS are excluded
from the means by default. Add `--metrics-include-eos` to include the true terminal
EOS in token metrics. Variant 5 averages only labeled frames, including every frame
of a token spanning sections, and never adds an artificial EOS frame.

Section sums and counts are combined before division, so a short section is not
weighted like a long section. Samples with no evaluated positions get `count=0`
and blank metric values. Sectioning uses the same video-context reset as training;
changing `--max-frames` can therefore change the predictions and metrics.

To produce **one combined CSV from existing checkpoints**, without retraining:

```bash
python3 tests/evaluate_checkpoints.py \
  --manifest data/train.jsonl \
  --checkpoints checkpoints/test_1_as_is.pt checkpoints/test_2_tcross_loss.pt \
    checkpoints/test_3_no_ht0.pt checkpoints/test_4_weighted_ht0.pt \
    checkpoints/test_5_frame_tokens.pt \
  --output results/all_variants.csv --batch-size 2
```

The evaluator uses each checkpoint's tokenizer, architecture, preprocessing,
saved AMP setting, and section size; `--max-frames` overrides that size, and
`--amp`/`--no-amp` overrides CUDA mixed precision. Its optional `--include-eos`
matches training's `--metrics-include-eos`. It also accepts `--device`, `--workers`,
and `--no-face-detector`. Older checkpoints can be evaluated with this script too.

## Resume and inference

```bash
python3 tests/train_4_weighted_ht0.py --manifest data/train.jsonl --resume checkpoints/test_4_weighted_ht0.pt --confidence-only --confidence-epochs 5
python3 infer.py --checkpoint checkpoints/test_4_weighted_ht0.pt --video clips/example.mp4
```

Fusion is applied consistently in token training, confidence-target generation,
and cached streaming inference. Checkpoints record model configuration, fixed
`w_t`, and training arguments. Older baseline checkpoints still load.

Resume restores the checkpoint's architecture and `w_t`; `--w-t` selects a
value for a **new** model. Use a matching variant checkpoint for variants 3/4/5;
resuming with a mismatched fusion raises an error. Variants 1/2 share an
architecture and can resume each other's checkpoints. As in the original trainer,
training hyperparameters come from the current invocation: repeat any customized
margin, loss coefficient, learning rate, batch size, `--amp`, `--max-frames`, or
`--no-face-detector` setting (or set them in `DEFAULTS`). Epoch counts request
additional epochs; optimizers, gradient-scaler state, and RNG state are not
restored. Further token training resets the confidence-trained status and its
epoch count, so refit confidence for the updated model.
`training_args` in the checkpoint provides a record of those settings.
For frame checkpoints, cached prediction returns the current frame's distribution;
`infer.py` does not perform an end-of-video autoregressive `--flush-tokens` pass.

Install `requirements-downvid.txt` for grouping dependencies, then run the
regression checks from the repository root:

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

For the runner's command-routing and CPU integration checks alone, use
`python3 -m unittest discover -s tests -p test_run_all_variants.py -v`.
