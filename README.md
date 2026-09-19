# CESM: causal video-to-text prototype

CESM is a research prototype for learning to transcribe speech from video of a
speaker's mouth. It pairs a causal video encoder with a text encoder, predicts
the next token from the available video and text history, and learns a per-frame
confidence signal to decide when to emit that token during streaming inference.

The repository supports training on your own aligned videos or preparing clips
from TalkVid or HDTF, experimenting with five fusion/loss variants, and running cached
inference on video files. The default text encoder is learned from scratch at
the character level; a Hugging Face causal language model can replace it.

- [Getting started](#getting-started): install, train, and transcribe a video.
- [Data and alignment](#data-and-alignment): prepare a JSONL training manifest.
- [Pretrained text encoder](#pretrained-text-encoder): use a frozen or fine-tuned LM.
- [Checkpoints and inference](#checkpoints-and-inference): load, resume, and train confidence.
- [Architecture experiments](#architecture-experiments): compare the five variants.
- [Download videos and train](#download-videos-and-train): prepare data, cache clips, and authenticate.
- [Training memory](#training-memory): configure AMP, video sections, and attention losses.
- [Troubleshooting](#troubleshooting): resolve download and alignment failures.
- [Repository guide](#repository-guide): find implementation, mathematics, and regression checks.

## How it works

The video path uses a fixed mouth crop, causal Conv3D frontend, per-frame residual
encoder, and causal Conformer-like stack. The text path encodes the preceding
token sequence. A fusion block combines their projected states for token
prediction and confidence estimation, with causal masks controlling which
frames and text states are available to each prediction.

Training has two stages:

1. Train token prediction (or frame prediction in variant 5) with configurable
   projection and attention regularizers. Only projection norm penalties have
   nonzero regularization weights by default.
2. Freeze the token prediction path and train confidence from the predictor's
   normalized entropy and future-distribution instability.

At inference, frames are processed one at a time and intermediate states are
cached. The neural network uses only the frames seen so far. See
[`train_mathematics.tex`](train_mathematics.tex) for the model equations,
alignment rules, losses, and two-stage training procedure.

## Getting started

Run these commands from the repository root in your Python environment. Install
the core training and inference dependencies:

```bash
python3 -m pip install -r requirements.txt
```

For an existing [aligned dataset](#data-and-alignment), train the default
character-level model and save its checkpoint:

```bash
python3 train.py \
  --manifest data/train.jsonl \
  --max-frames 256 \
  --batch-size 2 \
  --epochs 50 \
  --confidence-epochs 5 \
  --output checkpoints/cesm.pt
```

`data/train.jsonl` must describe your own videos and alignments. To create a
dataset from TalkVid or HDTF, follow [Download videos and train](#download-videos-and-train).
The trainer selects CUDA when available and otherwise uses CPU; `--device`
overrides that choice. On CUDA, add `--amp` to enable mixed precision.
`--max-frames` bounds the video section size used during training; its
[sectioning semantics](#video-sectioning) preserve transcript context.

Transcribe a video with the trained checkpoint:

```bash
python3 infer.py \
  --checkpoint checkpoints/cesm.pt \
  --video clips/example.mp4
```

Inference reads a video file incrementally and emits text as tokens are
committed. Add `--json-events` for structured events or `--pace-realtime` to pace
processing to the source frame rate when inference is faster than playback.
Use `python3 train.py --help` and `python3 infer.py --help` for all CLI options.

## Data and alignment

Training takes a JSONL manifest with one example per line:

```json
{"video":"clips/hello.mp4","text":"hello","windows":[[0,3],[4,6],[7,9],[10,12],[13,16]]}
```

Each row supplies a video path, its transcript, and inclusive, zero-based
`[start_frame, end_frame]` windows. Relative video paths resolve from the
manifest's directory. With the default character tokenizer, `windows[i]` aligns
the Unicode character `text[i]` to its frames. Windows must be ordered and
non-overlapping; gaps between them are allowed.

### Pretrained token alignment

When a pretrained LM is selected, these per-character windows are converted to
per-token windows using the fast tokenizer's offset mapping. A multi-character
subword receives the union of its character windows. If a byte-level tokenizer
uses several token IDs for one Unicode character, the character's frames are
split among those IDs. The character window must therefore contain at least as
many frames as the tokenizer produces IDs for that character.

Manifests that already contain one non-overlapping window per encoded LM token
are also accepted when their window count differs from the source-text length.
If the token and character counts happen to be equal, set
`"window_unit":"token"` on the row to remove the ambiguity. Use exactly the
same tokenizer/model named by `--pretrained-lm` when creating such alignments.

## Pretrained text encoder

Pass a Hugging Face decoder-only causal LM with a fast tokenizer to replace
the learned character encoder:

```bash
python3 train.py \
  --manifest data/train.jsonl \
  --pretrained-lm distilgpt2 \
  --output checkpoints/cesm-distilgpt2.pt
```

The imported LM is frozen by default, so CESM trains its video encoder, fusion
block, token head, and confidence path without updating the LM. To fine-tune
the LM too, add `--fine-tune-pretrained-lm`. A local model directory works in
place of the Hub model name; add `--pretrained-lm-local-files-only` to prohibit
network resolution.

`--d-text` and `--text-layers` apply only to the learned character encoder. With
`--pretrained-lm`, the text width and layer stack come from the imported model.

## Checkpoints and inference

When using a pretrained text encoder, CESM checkpoints include:

- the imported LM weights as part of `model_state`;
- the LM architecture config needed to rebuild it without downloading weights;
- the complete fast-tokenizer backend and CESM special-token mapping; and
- whether the imported LM was frozen.

Inference therefore needs the `transformers` and `tokenizers` packages, but it
does not need to download the original pretrained model:

```bash
python3 infer.py \
  --checkpoint checkpoints/cesm-distilgpt2.pt \
  --video clips/example.mp4
```

On CUDA, inference restores the checkpoint's training `--amp` setting for
both text and video streaming. Use `--amp` or `--no-amp` to override it. Older
checkpoints without a saved setting default to disabled; CPU inference also
disables AMP. `--json-events` reports the active AMP setting and dtype.

Token commitment uses the learned confidence head. Its threshold can relax as
frames pass since the previous commit, subject to a threshold floor, warmup,
minimum frame gap, and optional token-probability and stability guards. A
confidence-trained checkpoint is needed for meaningful commit timing; confidence
targets measure entropy and future instability rather than explicit boundaries.

`--repeat-token-cooldown-frames N` requires at least `N` frames since the same
token ID was last emitted (default: `5`; `0` disables the guard). For example,
with `N=5`, a token emitted at frame 10 can be emitted again starting at frame 15,
even if other tokens were emitted in between. Earlier predictions of that token
are skipped without selecting a runner-up or accumulating confidence/stability;
video processing continues. The optional `--flush-tokens` pass also respects this
cooldown and stops when its next token is blocked, since no more frames can pass.

### Resume training

Resume training with `--resume`. For a pretrained-LM checkpoint, the
stored architecture and tokenizer are authoritative, so `--pretrained-lm` may
be omitted; if supplied, its name must match the checkpoint. A checkpoint that
previously kept the LM frozen can be resumed with `--fine-tune-pretrained-lm`
to unfreeze it for the new token-training stage.

### Train only confidence

To load a checkpoint and train only its confidence stage:

```bash
python3 train.py \
  --manifest data/train.jsonl \
  --resume checkpoints/cesm-distilgpt2.pt \
  --confidence-only \
  --confidence-epochs 5 \
  --output checkpoints/cesm-distilgpt2-confidence.pt
```

`--confidence-only` requires `--resume` and a positive `--confidence-epochs`
count (default: 5). It overrides `--epochs` to zero, freezes the token path,
and trains only the confidence branch. The saved token epoch is preserved and
confidence epochs continue from the checkpoint's count. The existing
`--resume ... --epochs 0` form also remains supported.

## Architecture experiments

Compare five training variants with independent defaults in [`tests/`](tests/README.md):
the baseline, a per-token `tcross` norm-margin loss, fusion without the `ht0`
residual, fusion with a fixed configurable scalar multiplying `ht0`, and per-frame
token prediction from `text_ln(hv0 + vcross)`.
Each writes its own checkpoint and a per-sample CSV with mean true-class NLL and
probability margin. The
[experiment guide](tests/README.md#per-sample-csv-after-training) also describes
combining checkpoint results into one CSV without retraining.

The runner's evaluation stage and the standalone evaluation command require
`tests/evaluate_checkpoints.py`, which is not present in this checkout. Individual
variant scripts can still write their per-sample CSVs through the shared trainer.

To download TalkVid once, train all five variants, and evaluate their checkpoints
on those exact downloaded samples, use [`run_all_variants.py`](run_all_variants.py):

```bash
python3 -m pip install -r requirements-downvid.txt
python3 run_all_variants.py -N 20 --dataset-language English \
  --run-dir runs/talkvid_comparison --max-frames 256 \
  --batch-size 2 --lr 3e-4 --epochs 50 --w-t 0.5
```

FFmpeg (`ffmpeg` and `ffprobe`) and a supported yt-dlp JavaScript runtime must be
available as described in [dataset setup](#download-videos-and-train). Each variant
is evaluated as soon as its training finishes, saving a CSV beside its checkpoint and updating
`runs/talkvid_comparison/all_variants.csv` before the next variant starts.
See the [all-in-one runner instructions](tests/README.md#all-in-one-download-train-and-evaluate)
for per-variant overrides and reusing an existing manifest.

These CSV metrics describe the training samples with the true preceding text
supplied to the model. They are not held-out or free-running transcription
scores; see the experiment guide for metric definitions.

### Video projection and frame text contribution

Use `--no-vproj` to pass the video encoder's features directly into fusion.
The fusion layers then use `--d-video` as their width, and the text projection
maps from the text encoder's width to `--d-video`. This overrides `--d-fusion`
and skips the video projection similarity/norm penalties. It works with every
fusion variant; the default keeps the learned video projection.

For `--text-fusion frame`, `--vcross-weight W` sets the fixed scalar in
`text_ln(hv0 + W * vcross)` before the text FFN and token head. The default is
`1.0`; `0.0` makes token predictions depend only on video. For example:

```bash
python3 train.py --manifest data/train.jsonl \
  --text-fusion frame --no-vproj --vcross-weight 0.25 \
  --output checkpoints/frame-direct-video.pt
```

Confidence training uses token distributions with the same weight. The
confidence branch itself still attends to the causal text history. Both options
also work through `train_downvid.py`; `run_all_variants.py` applies `--no-vproj`
to all variants and routes `--vcross-weight` only to variant 5.

Checkpoints store both settings, and inference/resume restore them automatically.
Older checkpoints retain projection and weight `1.0`. Removing projection from
an existing projected checkpoint requires a fresh training run. When resuming,
omit these options or supply matching settings; an explicit different
`--vcross-weight` is rejected.

### Window phasing for frame targets

Use `--window-phasing 0.1` with `--text-fusion frame` (variant 5) to also supervise
the previous window's token during the first 10% of each new token window,
rounded down. The fraction must be in `[0,1]`; the default `0` disables phasing.

For a window of `L` frames, the last phased zero-based offset is
`floor(window_phasing * L) - 1`. A 20-frame window with `0.1` therefore phases
offsets 0 and 1; a window shorter than 10 frames has no phased frames.
Each phased frame contributes `0.5 * CE(current) + 0.5 * CE(previous)`.
The batch loss still averages over valid frames, so a phased frame counts once.
Repeated consecutive token IDs have the same loss as a single target.

The first token window has no previous target. Gaps and padding remain ignored.
Phasing uses the full original token window, including across `--max-frames`
boundaries; it never restarts at a section cut. Teacher-token availability and
confidence windows retain their existing boundaries. Evaluation CSVs continue
to score the current token, making them comparable across phasing settings.

```bash
python3 tests/train_5_frame_tokens.py --manifest data/train.jsonl --window-phasing 0.1
```

The flag also works in `train_downvid.py`; `run_all_variants.py` routes it only
to variant 5. The checkpoint records it in `training_args`. As with other loss
hyperparameters, pass the desired value again when resuming training.

## Download videos and train

`train_downvid.py` downloads **N usable videos/clips** from `--dataset talkvid`
(the default) or `--dataset hdtf`, uses
`grouper.py` to transcribe speech and group frames,
writes `data.jsonl`, then launches this repository's `train.py`. TalkVid's
[official download instructions](https://github.com/FreedomIntelligence/TalkVid/tree/main/data_pipeline/0_video_download)
use YouTube URLs and clip timestamps from the
[Hugging Face metadata](https://huggingface.co/datasets/FreedomIntelligence/TalkVid/tree/main/data).
The script streams `data/filtered_video_clips.json`; it does not download the
entire metadata array or TalkVid model weights. N counts clips, so several
clips may come from the same source video.

Install FFmpeg (both `ffmpeg` and `ffprobe` must be on PATH) and the optional
pipeline dependencies in your training environment. TalkVid also requires a
supported [JavaScript runtime](#downloader-runtime-talkvid-only):

```bash
python3 -m pip install -r requirements-downvid.txt

python3 train_downvid.py \
  --num-videos 100 \
  --dataset-language English \
  --work-dir data/talkvid \
  --max-frames 256 \
  --batch-size 2 \
  --epochs 50 \
  --output checkpoints/talkvid.pt
```

Every `train.py` argument is available, including `--pretrained-lm`, `--amp`,
`--flash-mono`, `--resume`, and all confidence-stage options. `--output` sets
the checkpoint path. `--manifest` is optional here and sets
the **generated** JSONL destination; it defaults to `WORK_DIR/data.jsonl`.
Relative CLI paths resolve from your current directory. With no `--work-dir`,
the default is `data/talkvid` or `data/hdtf` beside the script, according to
`--dataset`. The old `train_talkvid.py` command remains a compatibility entry point.

The grouping code defaults to `grouper.py` in the same folder as
`train_downvid.py`, independent of the current working directory. To use a
different implementation, pass `--group-frames-code /path/to/grouper.py`.
Directories containing `group_video_frames_by_token.py` are also accepted as an
explicit override. The wrapper imports the grouper's Python API without editing it.

### HDTF: hosted videos without YouTube

HDTF mode reads MP4 videos from the public
[Hugging Face HDTF mirror](https://huggingface.co/datasets/global-optima-research/HDTF).
It requests the ZIP index and selected members using HTTP byte ranges instead of
downloading the whole archive. No YouTube cookies, yt-dlp, or JavaScript runtime
are required for this mode; FFmpeg and faster-whisper are still required. The
default `videos.zip` contains 400 videos. N counts usable **source videos**, not
frame sections; some candidates may fail alignment, so 400 usable results are
not guaranteed. Videos are selected in filename order.

```bash
python3 train_downvid.py --dataset hdtf -N 100 \
  --download-max-frames 256 --language en \
  --max-frames 128 --batch-size 2 --epochs 50 \
  --output checkpoints/hdtf.pt

python3 run_all_variants.py --dataset hdtf -N 100 \
  --download-max-frames 256 --language en \
  --max-frames 128 --batch-size 2 --epochs 50 \
  --run-dir runs/hdtf_comparison
```

Add `--prepare-only` to the standalone downloader to create the manifest without
training. Use `--hdtf-archive /path/to/videos.zip` to work from a local archive,
or pass another HDTF ZIP URL supporting byte ranges. Each MP4 must include audio
for transcription and counts as one candidate. `--start-index` skips that many sorted videos;
`--max-attempts`, `--download-timeout`, `--download-retries`, alignment options,
and all training options remain available. `--dataset-language English` (or
`en`) is accepted; other language filters are rejected for HDTF.

**Frame limiting:** HDTF must transfer the complete compressed ZIP member before
trimming it locally. `--download-max-frames` caps the saved video and trims its
audio before transcription/grouping, but does not reduce the bytes
needed for that member. The untrimmed temporary file is removed afterward.
Different frame caps have separate caches. Full videos remain intact when the
option is omitted.

The downloader verifies each ZIP member CRC and the video/audio streams. Failed
media or alignment candidates are replaced until N are usable. Archive access
errors stop the run; servers that ignore byte ranges require a local archive.
Reruns reuse completed videos and alignments, but still read the archive index.

### Downloader runtime (TalkVid only)

YouTube extraction needs a supported JavaScript runtime and yt-dlp's EJS scripts.
The wrapper automatically enables installed runtimes; Node.js must be version
22 or newer. Runtime versions are checked before loading models or metadata;
an installed Node.js 20 does not satisfy this requirement. See the
[yt-dlp runtime setup guide](https://github.com/yt-dlp/yt-dlp/wiki/EJS).
Keep the downloader and its scripts current with
`python3 -m pip install -U 'yt-dlp[default]'`. Use `--js-runtimes node:/path/to/node`
to select a runtime outside PATH. Authentication can be supplied using
`--cookies FILE` or `--cookies-from-browser BROWSER` when needed.

### Pipeline options

| Option | Meaning |
| --- | --- |
| `--dataset talkvid` / `--dataset hdtf` | Select the source; TalkVid remains the default. |
| `--hdtf-archive PATH_OR_URL` | HDTF MP4 ZIP; defaults to the hosted `videos.zip`. |
| `--num-videos N` / `-N N` | Required number of successfully downloaded and aligned clips. |
| `--prepare-only` | Create the dataset without launching training. |
| `--metadata PATH_OR_URL` | TalkVid: override metadata with a JSON array or JSONL file/URL. |
| `--dataset-language English` | Filter metadata by language name; repeat to include several languages. |
| `--start-index N` | Skip N TalkVid metadata rows or N HDTF videos in filename order. |
| `--max-attempts N` | Limit eligible, distinct clips attempted; default is 10 times the requested count. |
| `--whisper-model small` | faster-whisper model name, size, or local directory. |
| `--language en` | Force the ASR language; omitted by default for automatic detection. |
| `--whisper-device cpu` | ASR device, independent of the trainer's `--device`. |
| `--whisper-compute-type int8` | ASR compute type; default is `default`. |
| `--cookies FILE` / `--cookies-from-browser firefox` | Optional yt-dlp authentication. |
| `--login` | Save YouTube cookies from the supplied file/browser (or guided prompt), then exit; no `-N` needed. |
| `--logout` | Delete the saved session for this work directory, then exit. |
| `--extractor-args STRING` | Optional yt-dlp extractor settings. |
| `--js-runtimes RUNTIME[:PATH]` | Override automatic detection of Deno, Node.js and QuickJS on PATH; repeatable. |
| `--download-format SELECTOR` | yt-dlp format selection; must retain both audio and video. |
| `--download-max-frames N` | Keep at most the first N frames of each clip and trim audio to match; default: full clip. |
| `--download-timeout 600` / `--download-retries 3` | Per-clip timeout in seconds and downloader retries. |
| `--image-ext jpg` / `--jpeg-quality 95` | Settings for the grouper's saved frame images. |
| `--reprocess` | Recompute frame alignments while reusing completed downloads. |

### Downloads and caching

TalkVid downloads use the metadata's time ranges, with re-encoding at cuts, and retain
audio for Whisper. The script checks the downloaded duration and both media
streams. Downloads and alignments are cached under `WORK_DIR/clips` and
`WORK_DIR/frames`; changes to the tokenizer, ASR settings, or grouper code create
a separate alignment cache. Frame images can occupy substantially more disk
space than the compressed clips. Metadata is read again on each run, so use a
local `--metadata` file when remote metadata access is unavailable.

### Limit clip length

To limit the downloaded and preprocessed video length, use for example:

```bash
python3 train_downvid.py -N 20 --download-max-frames 256 --prepare-only
```

The limit applies from each clip's start. Shorter clips stay whole. In TalkVid mode,
FFmpeg stops the video and audio during download, and the script verifies the
saved frame count before transcription. Network buffering and source keyframes
can require fetching extra data. Different limits have separate download and
alignment caches. Very short clips without usable speech are skipped as usual.
`--max-frames` remains the independent training-section size; it does not reduce
the amount downloaded. Omit `--download-max-frames` to download full clips.

### Incomplete runs and recovery

Unavailable videos and unusable alignments are logged in `WORK_DIR/run_report.json`
and replaced with later eligible clips until N are ready. The final JSONL contains
exactly those N clips, even if a previous run cached more. If metadata runs out or
the attempt limit is reached, the script exits without training, preserves any
previous final manifest, and saves the new partial rows as `data.jsonl.partial`
(or `MANIFEST.partial`). Raise `--max-attempts` or adjust the filter and rerun.
Per-download diagnostics are stored in `clips/.../download.log`.

When yt-dlp reports an unavailable or removed upload, the remaining metadata
clips from that upload are skipped for the current run without consuming more
attempts; valid cached clips can still be reused. Download errors include
yt-dlp's actual reason in the console and report. Bot checks, rate limits, and
recognized setup errors stop the run so you can correct them instead of spending
the entire attempt budget.

### Alignment and ASR

Alignment automatically follows the training tokenizer: one Unicode character
per group for the default learned encoder, or the exact fast tokenizer from
`--pretrained-lm` or a resumed checkpoint. Rows explicitly label `window_unit`
to avoid ambiguous character/token counts. The trainer's repair of
isolated zero-frame windows is checked before admitting a clip; irreparable
alignments are skipped. When resuming a learned character model, the prepared
transcripts must produce exactly the checkpoint's character vocabulary, as
required by `train.py`. Whisper runs in a separate process that exits before
training, releasing its GPU memory.

For example, prepare a Spanish dataset using CPU ASR and a pretrained tokenizer:

```bash
python3 train_downvid.py -N 20 \
  --dataset-language Spanish --language es \
  --pretrained-lm Qwen/Qwen3-0.6B \
  --whisper-device cpu --whisper-compute-type int8 \
  --work-dir data/talkvid-es --prepare-only
```

Then rerun without `--prepare-only` and add the desired training options, or
run `train.py --manifest data/talkvid-es/data.jsonl --pretrained-lm Qwen/Qwen3-0.6B`
directly. TalkVid's dataset is distributed under
[CC BY-NC 4.0](https://huggingface.co/datasets/FreedomIntelligence/TalkVid).

### Sign in locally or in Colab

YouTube authentication uses a signed-in browser's cookies. YouTube OAuth login
currently does not work with yt-dlp; see the
[official authentication guidance](https://github.com/yt-dlp/yt-dlp/wiki/Extractors#logging-in-with-oauth).

On the same computer as your browser, sign into YouTube there, then save its
session (replace `chrome` with `firefox` or another yt-dlp-supported browser):

```bash
python3 train_downvid.py --login --cookies-from-browser chrome --work-dir data/talkvid
```

`--login` alone offers a prompt for a cookie-file path or `browser:chrome`.
It imports an existing browser session; enter your password only on YouTube's
own sign-in page. A browser/profile specification such as `chrome:Profile 1`
is also supported. Saved sessions contain only unexpired YouTube-domain cookies.

**Colab runs on a remote machine**, so it cannot read the cookies from Chrome
on your laptop. Export YouTube cookies locally in Netscape format as `cookies.txt`
using the [YouTube cookie export instructions](https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies),
then upload that file in a notebook cell:

```python
from google.colab import files
uploaded = files.upload()  # Select cookies.txt; do not print its contents.
```

Run these notebook commands, adjusting the script path if necessary:

```python
!python /content/train_downvid.py --login --cookies /content/cookies.txt --work-dir /content/data/talkvid
!python /content/train_downvid.py -N 5 --download-max-frames 256 --work-dir /content/data/talkvid --prepare-only
```

The session is stored at `WORK_DIR/.auth/youtube.cookies.txt` with private file
permissions on POSIX and reused automatically for that work directory. Explicit
`--cookies` or `--cookies-from-browser` options override it. Rerun `--login` with
fresh cookies if the session expires; `--logout --work-dir ...` removes the saved
copy without signing your browser out or deleting your original export.
Keep these cookie files private; they grant access to your session.

Saving cookies validates their format and the presence of unexpired YouTube
sign-in cookies locally. It does not guarantee that YouTube accepts the session,
and a blocked Colab IP may still receive a bot-check error even with cookies.

## Training memory

### CUDA mixed precision

Pass `--amp` to enable CUDA mixed precision in both training stages and final
metrics. It also enables these memory reductions:

- Video crops and padded batches are stored in FP16 after FP32 pixel normalization.
- GroupNorm and LayerNorm calculate in FP32 and return activations in the active
  autocast dtype. Learned text embeddings use that same activation dtype.
- The visual frontend is activation-checkpointed during token training, trading
  recomputation during backward for lower memory use.
- Losses with zero coefficients are skipped and logged as zero. When monotonic
  and alignment coefficients are both zero, token attention uses SDPA without
  materializing attention weights or statistics.
- Large token/frame cross-entropy calculations use checkpointed chunks of 128
  positions. Confidence divergence temporaries use chunks of at most 128 future
  frames. Probability calculations and reductions remain FP32.
- A frozen FP32 pretrained LM uses the autocast parameter dtype. Already-half
  frozen LMs keep their existing dtype and numerical buffers keep their precision.
  Trainable LM parameters use FP32, including when resuming a frozen half-precision
  checkpoint with `--fine-tune-pretrained-lm`; trainable CESM weights and Adam
  moments also remain FP32.

Without `--amp`, FP32 execution and loss diagnostics are retained.
The CLI enables AMP only on CUDA. The standalone checkpoint evaluator inherits
each checkpoint's saved AMP setting; `--amp` or `--no-amp` overrides it. Full
vocabulary logits and each confidence window's probabilities still exist, so
batch size, vocabulary size, and window length continue to affect peak memory.

### Video sectioning

Pass `--max-frames N` to split every video into consecutive sections of at most
`N` frames, keeping the shorter final section. For example:

```bash
python3 train.py --manifest data/train.jsonl --max-frames 256 --batch-size 2
```

Each section is an independent batch element in both training stages, so
`--batch-size` still limits the number of elements and padded videos never
exceed `N` frames. Tokenization happens before splitting. Each token belongs to
the section containing its window's final frame; windows crossing a boundary
are clipped at that section's start and all windows are shifted to local frame
indices. Section boundaries do not add BOS or EOS targets. Each section encodes
the original transcript prefix (including its one original BOS), and predicts
only the tokens assigned to that section. Its first query uses the actual
preceding text state. Prior tokens provide context without being trained again
as targets. EOS is trained only at the original video's final frame, including
when that final section contains only trailing silence.

Interior sections without completed tokens have no token targets; batches
containing only those sections are skipped. Confidence and attention alignment
use every real token window, including the last token of a non-final section,
and exclude the true terminal EOS. Confidence can see prior transcript states
and only sees each new token starting on the frame after that token's window.

Video context still resets at each section. Text context is recomputed from the
full causal prefix, so text-encoder memory and work grow with transcript length;
the frame tensors remain bounded by N. The loader counts frames once per video
at startup, then loads only the requested section while preserving the original
video's fixed mouth crop. Checkpoints record these sectioning semantics.
Older checkpoints remain loadable, but learning these boundary semantics
requires token training; confidence-only training cannot update them.
`N` must be positive; omitting the option keeps full-video training.

### Confidence targets and attention

Confidence targets are evaluated one aligned token window at a time. This keeps
the [mathematical formulation's](train_mathematics.tex) entropy and
future-instability equations unchanged while avoiding a dense
`[batch, window, timestep, vocabulary]` allocation; vocabulary working
memory scales with the longest individual window instead. Attention paths that
do not expose weights use PyTorch scaled-dot-product attention (SDPA), which
selects a fused or memory-efficient backend when supported. By default, the
supervised text-to-video path retains explicit weights for its losses.

### Streamed attention regularizers

Pass `--flash-mono` to compute the monotonic loss directly from projected Q/K:

```bash
python3 train.py --manifest data/train.jsonl --flash-mono
```

This opt-in path streams small token/frame score blocks through an online
softmax, accumulates the expected frame per head, then averages the normalized
expectations across heads. Backward recomputes those blocks instead of saving
attention scores or probabilities. In addition to Q/K, saved loss statistics
scale as `O(batch * heads * text_tokens)`, rather than
`O(batch * heads * text_tokens * frames)`. FP16/BF16 inputs use FP32 accumulation.
The token output uses SDPA, and enabled `--lambda-align` supervision accumulates
window mass in the same streamed pass. Alignment computation is skipped when
its coefficient is zero with this flag.

Both standard and `--flash-mono` attention regularizers use normalized
**pre-dropout** probabilities, with the causal and padding masks applied.
Dropout is applied only when aggregating values for the token attention output;
it cannot artificially remove frame mass from the losses. For the same Q/K,
the losses and gradients match up to floating point rounding, including when
attention dropout is enabled. EOS exclusion and loss averaging are unchanged.

The statistics implementation uses portable PyTorch operations, not a fused
CUDA loss kernel, and supports first-order gradients. The boolean
visibility mask is still built for token SDPA; its backend and memory use
depend on the device, dtype, mask and dropout support. Other model attention
paths and compute complexity are unchanged. The flag is a training option with
no checkpoint architecture changes; specify it again when resuming training.

## Troubleshooting

### FFmpeg frame-limit compatibility

Frame limiting uses the older-compatible FFmpeg option `-vsync 0`. If a Colab
log reports `Unrecognized option 'fps_mode'`, replace its `train_downvid.py`
with the updated file and rerun; completed clips stay cached.

### Windows download timeouts

The downloader sets `-short_seek_size 1` on every FFmpeg input so it seeks
instead of reading through megabytes of unwanted media. It also applies a
30-second network timeout per input and stops the entire downloader process
tree when the per-clip timeout expires, avoiding leftover FFmpeg processes and
locked temporary files. This addresses a Windows/FFmpeg 8.1 failure where a
section download could spend its entire 600-second budget draining an audio
response before seeking to the clip timestamp. Raising `--download-timeout`
does not address that failure; use the current downloader implementation.

### Unicode frame paths

The frame grouper writes images using Python's Unicode-aware file handling,
supporting token directories such as `000000_ĠThis`. If an older copy reports
`Failed to write frame`, update `grouper.py` and restart; completed video downloads
remain cached and affected alignments are rebuilt automatically.

### YouTube says “The page needs to be reloaded”

This is a player/session failure, not proof that the upload was removed or that
your cookies expired. yt-dlp has previously fixed this symptom by updating its
[JavaScript challenge solver](https://github.com/yt-dlp/yt-dlp/pull/16231).
The wrapper stops on this error and preserves the report and cached clips.

Update the downloader **inside Colab**, then inspect its version and runtime:

```python
!python -m pip install -U "yt-dlp[default]"
!python -m yt_dlp --version
!node --version
```

Node.js must be 22 or newer; alternatively install Deno 2.3 or newer following
the [runtime setup guide](https://github.com/yt-dlp/yt-dlp/wiki/EJS).
For a Colab runtime with Node.js 20, install Deno using its
[official npm installation method](https://docs.deno.com/runtime/getting_started/installation/):

```python
!npm install -g deno
!deno --version
```

Then add `--js-runtimes deno` to the original training command (and to the
standalone check below, replacing `--js-runtimes node`). Existing saved cookies
will still be used with the same `--work-dir`.

Remove any custom `--extractor-args` while diagnosing. With Node.js available,
test one URL with the saved session before rerunning the dataset pipeline:

```python
!python -m yt_dlp --ignore-config --no-playlist --js-runtimes node --cookies /content/data/talkvid/.auth/youtube.cookies.txt --skip-download --print "%(id)s %(format_id)s" -- "https://www.youtube.com/watch?v=-04ZSRBGcsk"
```

This checks extraction and format selection; it does not download video bytes.
Ensure that the cookie path exists and matches the `--work-dir` used at login.
If it still fails, compare with the same test on your local computer. A failure
only in Colab may be tied to its network/IP, so a successful local check does
not establish that downloading from Colab will work. Refresh cookies only if
the session itself is rejected. An HF Hub `HF_TOKEN` warning is separate from
YouTube authentication and does not explain these player errors.

## Repository guide

| File or directory | Purpose |
| --- | --- |
| [`train.py`](train.py) | Shared model, tokenization, dataset loading, training stages, and per-sample metrics. |
| [`infer.py`](infer.py) | Cached, incremental inference and confidence-based token commitment. |
| [`train_downvid.py`](train_downvid.py) | TalkVid/HDTF download, alignment, caching, and training wrapper. |
| [`grouper.py`](grouper.py) | Speech transcription and grouping frames by character or LM token. |
| [`run_all_variants.py`](run_all_variants.py) | Shared-dataset orchestration for the five experiments. |
| [`flash_mono.py`](flash_mono.py) | Streamed monotonic and aligned-window attention statistics. |
| [`tests/`](tests/README.md) | Variant entry points, experiment reference, and regression checks. |
| [`train_mathematics.tex`](train_mathematics.tex) | Mathematical formulation of the architecture and training procedure. |

### Regression checks

Run the available offline pipeline regression checks from the repository root:

```bash
python3 -m unittest discover -s tests -p test_download_recovery.py -v
```

The checks cover download recovery and frame limits, plus grouping integration
with fixed ASR timestamps. Runner checks, including a tiny CPU training and
evaluation integration, are in `tests/test_run_all_variants.py`; the evaluation
integration also requires the missing `tests/evaluate_checkpoints.py` noted above:

```bash
python3 -m unittest discover -s tests -p test_run_all_variants.py -v
```
