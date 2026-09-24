"""Select and fetch hosted Shofo MP4s without downloading the entire dataset."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


SHOFO_REPO = "Shofo/shofo-talking-head-en"
SHOFO_URL = f"https://huggingface.co/datasets/{SHOFO_REPO}"
CHUNK_SIZE = 1024 * 1024
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


class HubRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(request, fp, code, msg, headers, newurl)
        old, new = urlparse(request.full_url), urlparse(newurl)
        if redirected is not None and (old.scheme, old.netloc) != (new.scheme, new.netloc):
            # Hugging Face redirects large files to signed CDN URLs. The Hub
            # bearer token belongs only on the original origin.
            redirected.remove_header("Authorization")
        return redirected


def copy_shofo_video(clip, output, timeout, retries):
    from huggingface_hub import get_token, hf_hub_url

    url = hf_hub_url(SHOFO_REPO, clip.filename, repo_type="dataset", revision=clip.revision)
    headers = {"User-Agent": "CESM-DownVid/1.0", "Accept-Encoding": "identity"}
    token = get_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    opener = build_opener(HubRedirectHandler())
    deadline = time.monotonic() + timeout
    for attempt in range(retries + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Shofo video transfer timed out after {timeout}s")
        try:
            with opener.open(Request(url, headers=headers), timeout=min(30, remaining)) as response, \
                    output.open("wb") as destination:
                if response.status != 200:
                    raise OSError("Shofo server did not return a complete file")
                count = 0
                while True:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Shofo video transfer timed out after {timeout}s")
                    chunk = response.read1(CHUNK_SIZE)
                    if not chunk:
                        break
                    destination.write(chunk)
                    count += len(chunk)
                    if count > clip.size:
                        raise OSError("Shofo video exceeds its listed size")
                if count != clip.size:
                    raise OSError("Incomplete Shofo video response")
            return
        except HTTPError as exc:
            status = exc.code
            exc.close()
            if status in (401, 403):
                raise ShofoAccessError(access_message()) from None
            if status not in RETRYABLE_STATUS or attempt == retries:
                raise OSError(f"Shofo video download failed (HTTP {status})") from None
        except (OSError, URLError, HTTPException) as exc:
            if attempt == retries:
                raise OSError(f"Shofo video transfer failed after {attempt + 1} attempts "
                              f"({type(exc).__name__}); retry the download") from None
        time.sleep(max(0, min(2 ** attempt, 4, deadline - time.monotonic())))
