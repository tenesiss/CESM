"""Select and fetch hosted Shofo MP4s without downloading the entire dataset."""

from __future__ import annotations

import contextlib
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


SHOFO_REPO = "Shofo/shofo-talking-head-en"
SHOFO_URL = f"https://huggingface.co/datasets/{SHOFO_REPO}"
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class ShofoAccessError(RuntimeError):
    """Dataset access/setup failures must stop preparation, not skip every clip."""


def access_message():
    return (f"Cannot access {SHOFO_REPO}. Request/accept access at {SHOFO_URL}, "
            "then authenticate with `hf auth login` or set HF_TOKEN to a read token "
            "for an approved account. Check --shofo-revision and token permissions.")


@dataclass(frozen=True)
class ShofoClip:
    revision: str
    filename: str
    size: int

    @property
    def key(self):
        value = [SHOFO_REPO, self.revision, self.filename, self.size]
        return hashlib.sha256(json.dumps(value).encode()).hexdigest()[:24]

    @property
    def source_key(self):
        # Shofo filenames are TikTok video IDs. Keep identity independent of
        # revision and frame cap so validation excludes the same source upload.
        name = PurePosixPath(self.filename).stem
        return f"tiktok:{name}" if name.isascii() and name.isdigit() else f"shofo:{self.filename}"


def iter_shofo(args):
    from huggingface_hub import HfApi, get_token

    api = HfApi(token=get_token())
    for attempt in range(args.download_retries + 1):
        try:
            info = api.dataset_info(SHOFO_REPO, revision=args.shofo_revision,
                                    files_metadata=True, timeout=args.download_timeout)
            break
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (401, 403, 404):
                raise ShofoAccessError(access_message()) from None
            if (status is not None and status not in RETRYABLE_STATUS) or attempt == args.download_retries:
                # Hub exceptions can contain signed URLs; expose only the type.
                raise ShofoAccessError(
                    f"Cannot list Shofo videos ({type(exc).__name__}); check network access "
                    "and --shofo-revision, then retry."
                ) from None
            time.sleep(min(2 ** attempt, 4))
    if not info.sha:
        raise ShofoAccessError("Shofo listing has no commit revision; cannot pin video downloads")
    videos = sorted((entry for entry in info.siblings or []
                     if entry.rfilename.startswith("videos/") and entry.rfilename.lower().endswith(".mp4")),
                    key=lambda entry: entry.rfilename)
    if not videos:
        raise ShofoAccessError("Shofo repository contains no videos/*.mp4 files")
    print(f"Shofo: {len(videos)} videos at {info.sha}; selecting in filename order", flush=True)
    for entry in videos:
        path = PurePosixPath(entry.rfilename)
        if ".." in path.parts or "\\" in entry.rfilename or entry.size is None or entry.size <= 0:
            raise ShofoAccessError("Shofo listing contains an invalid video path or size")
        yield {"id": path.stem, "info": {"Language": "English"},
               "clip": ShofoClip(info.sha, entry.rfilename, entry.size)}


def hub_download(clip, output, *, force_download=False):
    """Use the Hub's authenticated, Xet-capable client in a disposable staging area."""
    from huggingface_hub import get_token, hf_hub_download

    source = Path(hf_hub_download(
        SHOFO_REPO, clip.filename, repo_type="dataset", revision=clip.revision,
        token=get_token(), local_dir=output.parent / "hub", force_download=force_download,
    ))
    if source.stat().st_size != clip.size:
        raise OSError("Shofo video does not match its listed size")
    source.replace(output)


def hub_download_result(clip, output, *, force_download=False):
    # Hub/Xet exceptions can include signed URLs and tokens. Never serialize their
    # messages or tracebacks to the parent, clip logs, or run reports.
    try:
        hub_download(clip, output, force_download=force_download)
        return {"ok": True}
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return {"ok": False, "status": status, "error": type(exc).__name__}


def run_hub_download(clip, output, timeout, *, force_download=False):
    # A thread timeout cannot stop native Xet transfers. Isolate each transfer in
    # a process so subprocess.run kills and reaps it when its deadline expires.
    request = {"clip": clip.__dict__, "output": str(output.resolve()),
               "force_download": force_download}
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())], input=json.dumps(request),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        timeout=timeout, check=True,
    )
    return json.loads(result.stdout)


def copy_shofo_video(clip, output, timeout, retries):
    deadline = time.monotonic() + timeout
    force_download = False
    for attempt in range(retries + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Shofo video transfer timed out after {timeout}s")
        try:
            result = run_hub_download(clip, output, remaining, force_download=force_download)
            if result["ok"]:
                # Check again at the process boundary before committing a clip.
                if output.stat().st_size == clip.size:
                    return
                result = {"status": None, "error": "SizeMismatch"}
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Shofo video transfer timed out after {timeout}s") from None
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            result = {"status": None, "error": type(exc).__name__}
        status = result.get("status")
        if status in (401, 403):
            raise ShofoAccessError(access_message()) from None
        if status is not None and status not in RETRYABLE_STATUS:
            raise OSError(f"Shofo video download failed (HTTP {status})") from None
        if attempt == retries:
            raise OSError(f"Shofo video transfer failed after {attempt + 1} attempts "
                          f"({result.get('error', 'HubDownloadError')}); retry the download") from None
        # Do not accept an invalid completed local file from a previous attempt.
        # Other failures leave the Hub's partial data available for resumption.
        force_download = result.get("error") in ("OSError", "SizeMismatch")
        time.sleep(max(0, min(2 ** attempt, 4, deadline - time.monotonic())))


if __name__ == "__main__":
    request = json.load(sys.stdin)
    # Keep library diagnostics out of the JSON protocol (and credential-safe logs).
    with contextlib.redirect_stdout(sys.stderr):
        result = hub_download_result(ShofoClip(**request["clip"]), Path(request["output"]),
                                     force_download=request.get("force_download", False))
    json.dump(result, sys.stdout)
