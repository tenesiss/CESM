# CESM prototype

This repository contains a causal video-to-text trainer (`train.py`) and its
cached streaming inference path (`infer.py`). The original learned,
character-level text encoder remains the default.

## Architecture experiments

Five runnable training variants with independent defaults are in [`tests/`](tests/README.md):
the baseline, a per-token `tcross` norm-margin loss, fusion without the `ht0`
residual, fusion with a fixed configurable scalar multiplying `ht0`, and per-frame
token prediction from `text_ln(hv0 + vcross)`.
Each writes its own checkpoint and a per-sample CSV with mean true-class NLL and
probability margin. The same folder includes an evaluator for combining checkpoint
results into one CSV without retraining.

To download TalkVid once, train all five variants, and evaluate their checkpoints
on those exact downloaded samples, use [`run_all_variants.py`](run_all_variants.py):

```bash
python3 -m pip install -r requirements-talkvid.txt
python3 run_all_variants.py -N 20 --dataset-language English \
  --run-dir runs/talkvid_comparison --max-frames 256 \
  --batch-size 2 --lr 3e-4 --epochs 50 --w-t 0.5
```

FFmpeg (`ffmpeg` and `ffprobe`) and a supported yt-dlp JavaScript runtime must be
available as described below. The combined result is
`runs/talkvid_comparison/all_variants.csv`. See the [all-in-one runner instructions](tests/README.md#all-in-one-download-train-and-evaluate)
for per-variant overrides and reusing an existing manifest.

## Download TalkVid and train

`train_talkvid.py` downloads **N usable TalkVid clips**, runs the existing
`grouper.py` code to transcribe speech and group frames,
writes `data.jsonl`, then launches this repository's `train.py`. TalkVid's
[official download instructions](https://github.com/FreedomIntelligence/TalkVid/tree/main/data_pipeline/0_video_download)
use YouTube URLs and clip timestamps from the
[Hugging Face metadata](https://huggingface.co/datasets/FreedomIntelligence/TalkVid/tree/main/data).
The script streams `data/filtered_video_clips.json`; it does not download the
entire metadata array or TalkVid model weights. N counts clips, so several
clips may come from the same source video.

Install FFmpeg (both `ffmpeg` and `ffprobe` must be on PATH) and the optional
pipeline dependencies in your training environment:

```bash
python3 -m pip install -r requirements-talkvid.txt

python3 train_talkvid.py \
  --num-videos 100 \
  --dataset-language English \
  --work-dir data/talkvid \
  --max-frames 256 \
  --batch-size 2 \
  --epochs 50 \
  --output checkpoints/talkvid.pt
```

Every `train.py` argument is available, including `--pretrained-lm`, `--amp`,
`--flash-mono`, `--resume`, and all confidence-stage options. `--output` keeps
its training meaning (checkpoint path). `--manifest` is optional here and sets
the **generated** JSONL destination; it defaults to `WORK_DIR/data.jsonl`.
Relative CLI paths resolve from your current directory. With no `--work-dir`,
the default is `data/talkvid` beside the script.

The grouping code defaults to `grouper.py` in the same folder as
`train_talkvid.py`, independent of the current working directory. Place the
grouper implementation there, or pass `--group-frames-code /path/to/grouper.py`.
Directories containing `group_video_frames_by_token.py` are also accepted as an
explicit override. The wrapper imports the grouper's Python API without editing it.

Useful pipeline options:

| Option | Meaning |
| --- | --- |
| `--num-videos N` / `-N N` | Required number of successfully downloaded and aligned clips. |
| `--prepare-only` | Create the dataset without launching training. |
| `--metadata PATH_OR_URL` | Override the default metadata with a JSON array or JSONL file/URL. |
| `--dataset-language English` | Filter metadata by language name; repeat to include several languages. |
| `--start-index N` | Skip N raw metadata rows before applying language filters. |
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

Downloads use the metadata's time ranges, with re-encoding at cuts, and retain
audio for Whisper. The script checks the downloaded duration and both media
streams. Downloads and alignments are cached under `WORK_DIR/clips` and
`WORK_DIR/frames`; changes to the tokenizer, ASR settings, or grouper code create
a separate alignment cache. Frame images can occupy substantially more disk
space than the compressed clips. Metadata is read again on each run, so use a
local `--metadata` file when remote metadata access is unavailable.

To limit the downloaded and preprocessed video length, use for example:

```bash
python3 train_talkvid.py -N 20 --download-max-frames 256 --prepare-only
```

The limit applies from each metadata clip's start time. Shorter clips stay whole.
FFmpeg stops the video and audio during download, and the script verifies the
saved frame count before transcription. Network buffering and source keyframes
can require fetching extra data. Different limits have separate download and
alignment caches. Very short clips without usable speech are skipped as usual.
`--max-frames` remains the independent training-section size; it does not reduce
the amount downloaded. Omit `--download-max-frames` to download full clips.
Frame limiting uses the older-compatible FFmpeg option `-vsync 0`. If a Colab
log reports `Unrecognized option 'fps_mode'`, replace its `train_talkvid.py`
with the updated file and rerun; completed clips stay cached.

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

YouTube extraction needs a supported JavaScript runtime and yt-dlp's EJS scripts.
The wrapper automatically enables installed runtimes; Node.js must be version
22 or newer. Runtime versions are checked before loading models or metadata;
an installed Node.js 20 does not satisfy this requirement. See the
[yt-dlp runtime setup guide](https://github.com/yt-dlp/yt-dlp/wiki/EJS).
Keep the downloader and its scripts current with
`python3 -m pip install -U 'yt-dlp[default]'`. Use `--js-runtimes node:/path/to/node`
to select a runtime outside PATH. Authentication can be supplied using
`--cookies FILE` or `--cookies-from-browser BROWSER` when needed.

### Sign in locally or in Colab

YouTube authentication uses a signed-in browser's cookies. YouTube OAuth login
currently does not work with yt-dlp; see the
[official authentication guidance](https://github.com/yt-dlp/yt-dlp/wiki/Extractors#logging-in-with-oauth).

On the same computer as your browser, sign into YouTube there, then save its
session (replace `chrome` with `firefox` or another yt-dlp-supported browser):

```bash
python3 train_talkvid.py --login --cookies-from-browser chrome --work-dir data/talkvid
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
!python /content/train_talkvid.py --login --cookies /content/cookies.txt --work-dir /content/data/talkvid
!python /content/train_talkvid.py -N 5 --download-max-frames 256 --work-dir /content/data/talkvid --prepare-only
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

Alignment automatically follows the training tokenizer: one Unicode character
per group for the default learned encoder, or the exact fast tokenizer from
`--pretrained-lm` or a resumed checkpoint. Rows explicitly label `window_unit`
to avoid ambiguous character/token counts. The trainer's existing repair of
isolated zero-frame windows is checked before admitting a clip; irreparable
alignments are skipped. When resuming a learned character model, the prepared
transcripts must produce exactly the checkpoint's character vocabulary, as
required by `train.py`. Whisper runs in a separate process that exits before
training, releasing its GPU memory.

For example, prepare a Spanish dataset using CPU ASR and a pretrained tokenizer:

```bash
python3 train_talkvid.py -N 20 \
  --dataset-language Spanish --language es \
  --pretrained-lm Qwen/Qwen3-0.6B \
  --whisper-device cpu --whisper-compute-type int8 \
  --work-dir data/talkvid-es --prepare-only
```

Then rerun without `--prepare-only` and add the desired training options, or
run `train.py --manifest data/talkvid-es/data.jsonl --pretrained-lm Qwen/Qwen3-0.6B`
directly. TalkVid's dataset is distributed under
[CC BY-NC 4.0](https://huggingface.co/datasets/FreedomIntelligence/TalkVid).

Offline pipeline regression tests (the grouping integration tests use the local
grouper with fixed ASR timestamps and train a tiny CPU model):

```bash
python3 -m unittest test_train_talkvid
```

## Pretrained text encoder

Install the dependencies, then pass any Hugging Face decoder-only causal LM
with a fast tokenizer:

```bash
python3 -m pip install -r requirements.txt

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

`--d-text` and `--text-layers` apply only to the original learned encoder. With
`--pretrained-lm`, the text width and layer stack come from the imported model.

### Alignment and tokenization

The existing manifest format is still accepted:

```json
{"video":"clips/hello.mp4","text":"hello","windows":[[0,3],[4,6],[7,9],[10,12],[13,16]]}
```

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

### Checkpoints and inference

CESM checkpoints include:

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

Resume training normally with `--resume`. For a pretrained-LM checkpoint, the
stored architecture and tokenizer are authoritative, so `--pretrained-lm` may
be omitted; if supplied, its name must match the checkpoint. A checkpoint that
previously kept the LM frozen can be resumed with `--fine-tune-pretrained-lm`
to unfreeze it for the new token-training stage.

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

## Training memory

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

Without `--amp`, the original FP32 execution and loss diagnostics are retained.
The CLI enables AMP only on CUDA. The standalone checkpoint evaluator inherits
each checkpoint's saved AMP setting; `--amp` or `--no-amp` overrides it. Full
vocabulary logits and each confidence window's probabilities still exist, so
batch size, vocabulary size, and window length continue to affect peak memory.

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
video's fixed mouth crop. New checkpoints record these sectioning semantics.
Existing checkpoints remain loadable, but learning the corrected boundaries
requires token training; confidence-only training cannot update them.
`N` must be positive; omitting the option keeps full-video training.

Confidence targets are evaluated one aligned token window at a time. This keeps
the document's entropy and future-instability equations unchanged while avoiding
a dense `[batch, window, timestep, vocabulary]` allocation; vocabulary working
memory scales with the longest individual window instead. Attention paths that
do not expose weights use PyTorch scaled-dot-product attention (SDPA), which
selects a fused or memory-efficient backend when supported. By default, the
supervised text-to-video path retains explicit weights for its losses.

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
CUDA loss kernel, and supports first-order gradients. The existing boolean
visibility mask is still built for token SDPA; its backend and memory use
depend on the device, dtype, mask and dropout support. Other model attention
paths and compute complexity are unchanged. The flag is a training option with
no checkpoint architecture changes; specify it again when resuming training.
