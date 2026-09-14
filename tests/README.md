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
| `train_5_frame_tokens.py` | `text_ln(hv0 + vcross)` | Frame cross entropy replaces token cross entropy |

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
causally available text as keys/values. The text FFN and token head then produce
`[B, F, V]` logits. There is no `text_cross` module or `tcross` computation.
Each frame inside a supplied token window is labeled with that token. Unlabeled
frames and padding are excluded from both cross entropy and metrics; no EOS label
overwrites the last spoken token. Windows crossing a `--max-frames` boundary
supervise frames on both sides. The current token is hidden from attention until
the frame after its window ends, preserving the existing causal teacher forcing.

Projection losses retain their defaults. `--lambda-tcross`, `--lambda-mono`, and
`--lambda-align` must remain zero in variant 5 because there is no text-to-video
attention. `--flash-mono` has no effect on this variant. Its confidence stage
also freezes `video_cross`, which now participates in token prediction, and
uses frame distributions within the labeled windows for its targets.

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

## All-in-one download, train and evaluate

From the repository root, run `run_all_variants.py` to download and align TalkVid
**once**, train variants 1–5 sequentially, then load all five saved checkpoints
and evaluate them on **the exact same manifest samples**:

```bash
python3 -m pip install -r requirements-talkvid.txt
python3 run_all_variants.py \
  -N 20 --dataset-language English \
  --work-dir data/talkvid --run-dir runs/talkvid_comparison \
  --download-max-frames 256 --max-frames 128 \
  --epochs 50 --confidence-epochs 5 --batch-size 2 --lr 3e-4 \
  --tcross-margin 1.0 --lambda-tcross 1.0 --w-t 0.5
```

Install FFmpeg (`ffmpeg` and `ffprobe`) and a supported yt-dlp JavaScript runtime
as described in the repository README. All TalkVid download/alignment options
are available, including repeated `--dataset-language`, `--metadata`,
`--whisper-model`, `--whisper-device`, `--language`, `--cookies`,
`--cookies-from-browser`, `--js-runtimes`, and `--download-max-frames`.
The original TalkVid downloader reuses its existing clip/alignment cache and
replaces unusable candidates until it has N usable samples. An incomplete or
failed preparation stops the runner before any variant is trained.

To reuse an already prepared dataset without downloading or loading Whisper:

```bash
python3 run_all_variants.py \
  --manifest data/talkvid/data.jsonl --run-dir runs/existing_samples \
  --epochs 10 --batch-size 2 --max-frames 128 --w-t 0.5
```

Choose either `-N` or `--manifest`. `--work-dir` controls the reusable TalkVid
cache; `--run-dir` holds this experiment's outputs and must be empty or new.
If omitted, a new timestamped directory under `runs/` is used. Paths passed on
the CLI resolve from the current directory, so invoking the runner by its
absolute path from elsewhere also works.

The runner creates:

```text
RUN_DIR/
  samples.jsonl                    # Fixed sample order, labels and absolute video paths
  talkvid.jsonl                    # Downloader output (TalkVid mode only)
  checkpoints/test_1_as_is.pt
  checkpoints/test_2_tcross_loss.pt
  checkpoints/test_3_no_ht0.pt
  checkpoints/test_4_weighted_ht0.pt
  checkpoints/test_5_frame_tokens.pt
  all_variants.csv                 # Five rows per sample, one for each variant
  run.json                        # Manifest SHA-256, stage status, commands and output paths
  logs/download.log               # TalkVid mode only
  logs/train_1.log ... train_5.log
  logs/evaluate.log
```

`samples.jsonl` is copied once from the completed preparation (or input manifest),
with absolute video paths. Every training command and the final evaluator use
that snapshot, and its SHA-256 is checked between stages. The video files remain
in their original/download locations. Keep those files for later evaluation.
Training processes exit before the next stage starts, releasing Whisper and
model memory between stages. The runner disables the individual scripts'
automatic CSV passes and evaluates once, after all five checkpoints are saved.

Unspecified settings retain each variant script's `DEFAULTS`. Shared flags such
as `--lr`, `--batch-size`, `--epochs`, `--pretrained-lm`, `--amp`, and `--seed`
apply to all variants. Experiment-specific options route as follows:

| Shared CLI option | Variants receiving it |
| --- | --- |
| `--tcross-margin`, `--lambda-tcross` | 2 only |
| `--w-t` | 4 only |
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
`--metrics-include-eos` applies consistently to the final token metrics.

The combined CSV has the per-sample NLL and probability-margin columns described
below: token averages for variants 1–4, labeled-frame averages for variant 5.
These are measurements on the downloaded **training samples**. For N samples,
the CSV contains 5 × N data rows, even when videos are split into sections.
If a stage fails, later stages do not run; completed checkpoints, logs and the
stage status remain available. Once all checkpoints exist, the standalone
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
and no regularization terms. Metrics describe the training samples, not a held-out
test set or free-running transcription. For variants 1–4, each transcript token
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

The evaluator uses each checkpoint's tokenizer, architecture, preprocessing and
saved section size; `--max-frames` overrides that size. Its optional `--include-eos`
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
margin, loss coefficient, learning rate or batch size (or set them in `DEFAULTS`).
`training_args` in the checkpoint provides a record of those settings.
For frame checkpoints, cached prediction returns the current frame's distribution;
`infer.py` does not perform an end-of-video autoregressive `--flush-tokens` pass.

Run the offline regression checks with:

```bash
python3 -m unittest tests.test_arch_variants tests.test_frame_metrics tests.test_run_all_variants
```
