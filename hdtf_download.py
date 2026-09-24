"""Read selected videos from an HDTF ZIP using HTTP ranges or a local archive."""

from __future__ import annotations

import contextlib
import io
import re
import time
import zipfile
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DEFAULT_HDTF_ARCHIVE = (
    "https://huggingface.co/datasets/global-optima-research/HDTF/resolve/main/videos.zip"
)
CHUNK_SIZE = 1024 * 1024


class ArchiveAccessError(RuntimeError):
    """Archive-wide access failures should stop preparation, not skip every video."""


class HTTPRangeReader(io.RawIOBase):
    """Seekable ZIP input that never falls back to downloading the whole archive."""

    def __init__(self, url, timeout, retries, *, identity=None):
        super().__init__()
        self.url, self.timeout, self.retries = url, timeout, retries
        self.position = 0
        self.size, self.etag = identity or (None, None)
        self.deadline = time.monotonic() + timeout
        self._fetch(0, 0)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence not in (io.SEEK_SET, io.SEEK_CUR, io.SEEK_END):
            raise ValueError("Invalid seek origin")
        target = offset + (0 if whence == io.SEEK_SET else
                           self.position if whence == io.SEEK_CUR else self.size)
        if target < 0:
            raise ValueError("Cannot seek before archive start")
        self.position = target
        return target

    def _fetch(self, start, end):
        for attempt in range(self.retries + 1):
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise ArchiveAccessError(f"HDTF archive transfer timed out after {self.timeout}s")
            try:
                request = Request(self.url, headers={
                    "Range": f"bytes={start}-{end}", "Accept-Encoding": "identity",
                    "User-Agent": "CESM-DownVid/1.0",
                })
                with urlopen(request, timeout=min(30, remaining)) as response:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                         response.headers.get("Content-Range", ""))
                    if (response.status != 206 or not match or
                            (int(match[1]), int(match[2])) != (start, end)):
                        raise ArchiveAccessError(
                            "HDTF server did not honor HTTP byte ranges. "
                            "Use --hdtf-archive /path/to/videos.zip for a local archive."
                        )
                    size, etag = int(match[3]), response.headers.get("ETag")
                    if self.size is not None and (size != self.size or
                                                  (self.etag and etag != self.etag)):
                        raise ArchiveAccessError("HDTF archive changed during download; rerun preparation")
                    payload = response.read(end - start + 2)
                    if len(payload) != end - start + 1:
                        raise OSError("Incomplete HTTP range response")
                    self.size, self.etag = size, etag
                    return payload
            except HTTPError as exc:
                status = exc.code
                exc.close()
                if status not in (408, 429, 500, 502, 503, 504) or attempt == self.retries:
                    raise ArchiveAccessError(
                        f"Cannot access HDTF archive (HTTP {status}). Check --hdtf-archive "
                        "or supply a local ZIP. No YouTube cookies are used."
                    ) from None
            except (OSError, URLError, HTTPException) as exc:
                if attempt == self.retries:
                    # Do not print redirect URLs, which can contain signed credentials.
                    raise ArchiveAccessError(
                        f"HDTF archive network read failed after {attempt + 1} attempts "
                        f"({type(exc).__name__}); retry or supply a local ZIP."
                    ) from None
            time.sleep(max(0, min(2 ** attempt, 4, self.deadline - time.monotonic())))

    def read(self, size=-1):
        if self.closed:
            raise ValueError("Read from closed archive")
        end = self.size if size is None or size < 0 else min(self.size, self.position + size)
        chunks = []
        while self.position < end:
            stop = min(end, self.position + CHUNK_SIZE)
            chunks.append(self._fetch(self.position, stop - 1))
            self.position = stop
        return b"".join(chunks)


@contextlib.contextmanager
def open_archive(source, timeout, retries, *, identity=None):
    with contextlib.ExitStack() as stack:
        if urlparse(source).scheme in ("http", "https"):
            stream = stack.enter_context(HTTPRangeReader(source, timeout, retries, identity=identity))
        else:
            stream = stack.enter_context(Path(source).expanduser().open("rb"))
        try:
            archive = stack.enter_context(zipfile.ZipFile(stream))
        except zipfile.BadZipFile as exc:
            raise ArchiveAccessError("HDTF source is not a readable ZIP archive") from exc
        yield archive


def copy_video(archive, member, output, timeout):
    """CRC-check one member and write to a caller-chosen path, ignoring ZIP paths."""
    deadline = time.monotonic() + timeout
    if isinstance(archive.fp, HTTPRangeReader):
        archive.fp.deadline = deadline
    with archive.open(member) as source, output.open("wb") as destination:
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"HDTF video transfer timed out after {timeout}s")
            chunk = source.read(CHUNK_SIZE)
            if not chunk:
                break
            destination.write(chunk)
