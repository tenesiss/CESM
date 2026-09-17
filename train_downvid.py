#!/usr/bin/env python3
"""Download TalkVid or HDTF videos, align them with the local frame grouper, then train.

All training options come directly from train.build_argparser(). See --help
and README.md for download/preprocessing options. N counts usable clips, not
unique source YouTube uploads. HDTF uses hosted videos without YouTube access.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from hdtf_download import DEFAULT_HDTF_ARCHIVE, ArchiveAccessError, copy_video, open_archive


ROOT = Path(__file__).resolve().parent
DEFAULT_METADATA = (
    "https://huggingface.co/datasets/FreedomIntelligence/TalkVid/resolve/main/"
    "data/filtered_video_clips.json"
)
DEFAULT_GROUPER = ROOT / "grouper.py"
DEFAULT_FORMAT = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def build_argparser():
    import train

    parser = train.build_argparser()
    parser.description = __doc__.split("\n\n")[0]
    parser.allow_abbrev = False
    for action in parser._actions:
        if action.dest == "manifest":
            action.required = False
            action.help = "Generated JSONL path (default: WORK_DIR/data.jsonl)"
    group = parser.add_argument_group("Dataset download and preprocessing")
    group.add_argument("--dataset", choices=["talkvid", "hdtf"], default="talkvid",
                       help="Video source (default: talkvid); hdtf downloads hosted MP4s without YouTube")
    group.add_argument("--num-videos", "-N", type=positive_int,
                       help="Number of successfully aligned videos/clips to train on; required "
                            "except with --login or --logout")
    group.add_argument("--work-dir", type=Path,
                       help="Download/alignment cache (default: data/DATASET beside this script)")
    group.add_argument("--hdtf-archive", default=DEFAULT_HDTF_ARCHIVE,
                       help="HDTF ZIP URL or local ZIP path; remote servers must support HTTP byte ranges")
    group.add_argument("--metadata", default=DEFAULT_METADATA,
                       help="TalkVid JSON array or JSONL: local path or HTTP(S) URL")
    group.add_argument("--dataset-language", action="append", default=[], metavar="NAME",
                       help="Filter info.Language, e.g. English or Spanish; repeatable")
    group.add_argument("--start-index", type=int, default=0,
                       help="Skip N metadata rows or HDTF videos in filename order (default: 0)")
    group.add_argument("--max-attempts", type=positive_int,
                       help="Maximum eligible, distinct clips to try; remaining clips from an "
                            "unavailable upload are skipped without attempts (default: 10 * N)")
    group.add_argument("--download-timeout", type=positive_int, default=600,
                       help="Per-clip download timeout in seconds")
    group.add_argument("--download-retries", type=int, default=3)
    group.add_argument("--download-format", default=DEFAULT_FORMAT,
                       help="yt-dlp format selector; must include video AND audio")
    group.add_argument("--download-max-frames", type=positive_int, metavar="N",
                       help="Keep at most the first N frames of each downloaded clip, with matching "
                            "audio (default: full clip; independent of training --max-frames)")
    cookies = group.add_mutually_exclusive_group()
    cookies.add_argument("--cookies", type=Path, help="yt-dlp Netscape cookies file")
    cookies.add_argument("--cookies-from-browser", help="yt-dlp browser/profile specification")
    session = group.add_mutually_exclusive_group()
    session.add_argument("--login", action="store_true",
                         help="Save YouTube sign-in cookies for this work directory, then exit. "
                              "Use --cookies FILE or --cookies-from-browser BROWSER, or follow the prompt")
    session.add_argument("--logout", action="store_true",
                         help="Remove this work directory's saved YouTube cookies, then exit")
    group.add_argument("--extractor-args", help="yt-dlp extractor arguments")
    group.add_argument("--js-runtimes", action="append", default=[], metavar="RUNTIME[:PATH]",
                       help="yt-dlp JavaScript runtime; repeatable. Default: auto-detect Deno, "
                            "Node.js and QuickJS on PATH")
    group.add_argument("--group-frames-code", type=Path, default=DEFAULT_GROUPER,
                       help="Grouper Python file (default: grouper.py beside this script), or "
                            "directory containing group_video_frames_by_token.py")
    group.add_argument("--whisper-model", default="small")
    group.add_argument("--language", default=None,
                       help="Force Whisper language code (en, es, ...); default: auto-detect")
    group.add_argument("--whisper-device", choices=["auto", "cpu", "cuda"], default="auto",
                       help="Whisper device, independent of training --device")
    group.add_argument("--whisper-compute-type", default="default")
    group.add_argument("--image-ext", choices=["jpg", "jpeg", "png"], default="jpg")
    group.add_argument("--jpeg-quality", type=int, choices=range(1, 101), default=95,
                       metavar="1..100")
    group.add_argument("--reprocess", action="store_true",
                       help="Recompute alignments even when a matching cached result exists")
    group.add_argument("--prepare-only", action="store_true",
                       help="Download and generate data.jsonl, then exit without training")
    return parser


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_cache(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def login_cookie_path(args):
    return (args.work_dir or ROOT / "data" / args.dataset).expanduser().resolve() / ".auth" / "youtube.cookies.txt"


def prompt_login_source(args):
    print("Sign into YouTube in your browser first.\n"
          "On this computer, enter browser:chrome or browser:firefox to use that browser's session.\n"
          "For Colab/remote machines, upload an exported YouTube cookies.txt file and enter its path.\n"
          "Cookie export instructions: https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies\n"
          "Enter a file path or browser name, never your password or cookie values.", flush=True)
    # Notebook %run can use the kernel's input support; !python launches a
    # separate process without that input widget.
    try:
        from IPython import get_ipython
        notebook = get_ipython() is not None
    except ImportError:
        notebook = False
    if not sys.stdin.isatty() and not notebook:
        raise RuntimeError("Interactive login is unavailable here. After uploading cookies.txt, run "
                           "--login --cookies /content/cookies.txt (or use --cookies-from-browser "
                           "on the computer where you signed in).")
    try:
        source = input("Cookie file path or browser:BROWSER: ").strip()
    except EOFError:
        raise RuntimeError("No login source provided; use --login --cookies FILE.") from None
    if not source:
        raise RuntimeError("No login source provided; use --login --cookies FILE.")
    if source.startswith("browser:"):
        args.cookies_from_browser = source[len("browser:"):]
        if not args.cookies_from_browser:
            raise RuntimeError("Specify a browser, for example browser:chrome.")
    else:
        args.cookies = Path(source).expanduser()


def login(args):
    """Save locally imported YouTube cookies without contacting YouTube."""
    try:
        import yt_dlp
        from yt_dlp.cookies import YoutubeDLCookieJar
    except ImportError:
        raise RuntimeError(f"Login requires yt-dlp: {sys.executable} -m pip install -U 'yt-dlp[default]'") from None
    if not args.cookies and not args.cookies_from_browser:
        prompt_login_source(args)
    # Cookie parsers can include malformed cookie values in warnings/errors.
    # Suppress those diagnostics and expose only a safe, actionable message.
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if args.cookies:
                jar = YoutubeDLCookieJar(str(args.cookies.expanduser()))
                jar.load()
            else:
                options = yt_dlp.parse_options([
                    "--ignore-config", "--cookies-from-browser", args.cookies_from_browser,
                ]).ydl_opts
                with yt_dlp.YoutubeDL(options) as ydl:
                    jar = ydl.cookiejar
    except (Exception, SystemExit):
        raise RuntimeError("Cannot load sign-in cookies. Use a readable Netscape cookies.txt file, "
                           "or a supported browser/profile on this computer. In Colab, upload a "
                           "cookie file exported from your local browser and use --login --cookies FILE.") from None
    selected = YoutubeDLCookieJar()
    for cookie in jar:
        domain = cookie.domain.lstrip(".").lower()
        if (domain == "youtube.com" or domain.endswith(".youtube.com")) and not cookie.is_expired():
            selected.set_cookie(cookie)
    auth_names = {"SID", "SAPISID", "__Secure-1PAPISID", "__Secure-3PAPISID",
                  "__Secure-1PSID", "__Secure-3PSID"}
    if not any(cookie.name in auth_names and cookie.value for cookie in selected):
        raise RuntimeError("No unexpired YouTube sign-in cookies found. Sign into YouTube in the "
                           "selected browser/profile, or export fresh YouTube cookies and try again.")
    destination = login_cookie_path(args)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # NamedTemporaryFile creates a private file (0600 on POSIX). Replace only
    # after validation, retaining a previous saved session if import fails.
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        # yt-dlp serializes session cookies with expiry 0 before saving; keep
        # those entries. Actual expired cookies were already filtered above.
        selected.save(str(temporary), ignore_discard=True, ignore_expires=True)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved YouTube session cookies: {destination}\n"
          "Future runs with this --work-dir will use them automatically.\n"
          "Cookies were validated locally; YouTube access has not been checked. "
          "A blocked notebook IP can still be rejected.\n"
          "Keep the cookie file private. Use --logout to remove the saved copy.", flush=True)


def use_saved_login(args):
    saved = login_cookie_path(args)
    if not args.cookies and not args.cookies_from_browser and saved.is_file():
        args.cookies = saved
        print(f"Using saved YouTube session: {saved}", flush=True)


def iter_metadata(source):
    """Stream a JSON array or JSONL metadata file from a local path or HTTP(S)."""
    import ijson

    if urlparse(source).scheme in ("http", "https"):
        stream = urlopen(Request(source, headers={"User-Agent": "CESM-TalkVid/1.0"}), timeout=60)
    else:
        stream = Path(source).expanduser().open("rb")
    with stream, io.BufferedReader(stream) as buffered:
        # Consume an optional UTF-8 BOM and leading whitespace, including when
        # the stream is non-seekable (HTTP).
        if buffered.peek(3).startswith(b"\xef\xbb\xbf"):
            buffered.read(3)
        while buffered.peek(1)[:1] in (b" ", b"\t", b"\r", b"\n"):
            buffered.read(1)
        if buffered.peek(1).startswith(b"["):
            yield from ijson.items(buffered, "item", use_float=True)
        else:
            for line in buffered:
                if line.strip():
                    yield json.loads(line)


@dataclass(frozen=True)
class Clip:
    url: str
    start: float
    end: float

    @classmethod
    def from_row(cls, row):
        info = row.get("info") or {}
        if not isinstance(info, dict):
            raise ValueError("info must be a JSON object")
        url = info.get("Video Link") or info.get("video_link")
        if not isinstance(url, str) or urlparse(url).scheme not in ("http", "https"):
            raise ValueError("Missing HTTP(S) info.Video Link")
        start = float(row.get("start-time", row.get("start")))
        end = float(row.get("end-time", row.get("end")))
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError("Clip times must be finite and satisfy 0 <= start < end")
        return cls(url, start, end)

    @property
    def key(self):
        # Metadata IDs can contain path separators and need not be unique.
        return digest([self.url, self.start, self.end])[:24]

    @property
    def source_key(self):
        """Identify an upload across clip ranges and common YouTube URL forms."""
        parsed = urlparse(self.url)
        host = (parsed.hostname or "").lower()
        video_id = None
        if host in ("youtu.be", "www.youtu.be"):
            video_id = parsed.path.strip("/").split("/")[0]
        elif host in ("youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"):
            video_id = parse_qs(parsed.query).get("v", [None])[0]
            parts = parsed.path.strip("/").split("/")
            if not video_id and len(parts) == 2 and parts[0] in ("shorts", "embed", "live"):
                video_id = parts[1]
        if video_id and re.fullmatch(r"[\w-]{11}", video_id, flags=re.ASCII):
            return f"youtube:{video_id}"
        return self.url


class SourceUnavailableError(RuntimeError):
    """The source upload is unavailable, including its other metadata clips."""


class DownloadSetupError(RuntimeError):
    """A downloader configuration or access problem requires intervention."""


@dataclass(frozen=True)
class ArchiveClip:
    url: str
    member: str
    crc: int
    size: int

    @property
    def key(self):
        return digest(["hdtf", self.url, self.member, self.crc, self.size])[:24]

    @property
    def source_key(self):
        return self.key


def iter_hdtf(args):
    with open_archive(args.hdtf_archive, args.download_timeout, args.download_retries) as archive:
        videos = sorted((member for member in archive.infolist()
                         if not member.is_dir() and member.filename.lower().endswith(".mp4")
                         and not member.filename.startswith("__MACOSX/")),
                        key=lambda member: member.filename)
        if not videos:
            raise ArchiveAccessError("HDTF ZIP contains no MP4 videos")
        print(f"HDTF archive: {len(videos)} videos; selecting in filename order", flush=True)
        for member in videos:
            yield {"id": member.filename, "info": {"Language": "English"},
                   "clip": ArchiveClip(args.hdtf_archive, member.filename, member.CRC, member.file_size),
                   "archive": archive, "member": member}


def download_hdtf_clip(clip, args, archive, member):
    cached = cached_download(clip, args)
    if cached is not None:
        print(f"  Reusing download: {cached}", flush=True)
        return cached
    folder = download_folder(clip, args)
    folder.mkdir(parents=True, exist_ok=True)
    video = folder / "video.mp4"
    log = folder / "download.log"
    with tempfile.TemporaryDirectory(prefix="download-", dir=folder) as staging:
        source = Path(staging) / "source.mp4"
        with log.open("w", encoding="utf-8") as stream:
            stream.write(f"HDTF member: {clip.member}\n")
            stream.flush()
            try:
                copy_video(archive, member, source, args.download_timeout)
                duration = probe_video(source, None)
                output = source
                if args.download_max_frames is not None:
                    output = Path(staging) / "trimmed.mp4"
                    command = [
                        "ffmpeg", "-v", "error", "-y", "-i", str(source),
                        "-map", "0:v:0", "-map", "0:a:0",
                        "-vf", f"trim=end_frame={args.download_max_frames},setpts=PTS-STARTPTS",
                        "-af", "asetpts=PTS-STARTPTS", "-vsync", "0", "-shortest",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-c:a", "aac",
                        str(output),
                    ]
                    run_download_process(command, stdout=stream, timeout=args.download_timeout)
                    probe_video(output, duration, max_frames=args.download_max_frames)
                output.replace(video)
            except (OSError, ValueError, RuntimeError, zipfile.BadZipFile,
                    subprocess.SubprocessError) as exc:
                stream.write(f"Failed: {exc}\n")
                raise
    write_json(folder / "download.json", {"dataset": "hdtf", "clip": clip.__dict__,
               "max_frames": args.download_max_frames, "video_signature": video_signature(video)})
    return video


def download_runtimes(args):
    if args.js_runtimes:
        return args.js_runtimes
    runtimes = []
    for runtime, binaries in (("deno", ("deno",)), ("node", ("node", "nodejs")),
                              ("quickjs", ("qjs",))):
        for binary in binaries:
            path = shutil.which(binary)
            if path:
                runtimes.append(f"{runtime}:{path}")
                break
    return runtimes


def validate_download_runtimes(args):
    """Keep supported runtimes; fail before preparation if none are usable."""
    supported, descriptions, rejected = [], [], []
    requirements = {
        "node": ("node", r"^v(\S+)", (22, 0, 0)),
        "deno": ("deno", r"^deno (\S+)", (2, 3, 0)),
        "quickjs": ("qjs", r"^QuickJS(?:-ng)?(?: version)?\s+(\S+)", (2023, 12, 9)),
        "bun": ("bun", r"^(\S+)", (1, 2, 11)),
    }
    for specification in download_runtimes(args):
        name, _, path = specification.partition(":")
        if name not in requirements:
            raise DownloadSetupError(f"Unsupported JavaScript runtime: {name}")
        binary, pattern, minimum = requirements[name]
        executable = str(Path(path).expanduser()) if path else (shutil.which(binary) or binary)
        if Path(executable).is_dir():
            executable = str(Path(executable) / binary)
        try:
            result = subprocess.run(
                [executable, "--help" if name == "quickjs" else "--version"],
                capture_output=True, text=True, timeout=10,
            )
            output = (result.stdout or "") + (result.stderr or "")
            match = re.search(pattern, output, re.M)
            version = match.group(1) if match else "unknown"
            numbers = tuple(int(part) for part in re.findall(r"\d+", version)[:3])
            if name == "quickjs" and output.startswith("QuickJS-ng"):
                minimum = (0, 0, 1)
            valid = (result.returncode == 0 or name == "quickjs") and numbers >= minimum
            if name == "bun" and numbers > (1, 3, 14):
                valid = False
        except (OSError, subprocess.SubprocessError):
            version, valid = "unavailable", False
        if valid:
            supported.append(f"{name}:{executable}")
            descriptions.append(f"{name} {version}")
        else:
            rejected.append(f"{name} {version}")
    if not supported:
        found = ", ".join(rejected) or "none on PATH"
        raise DownloadSetupError(
            f"No supported YouTube JavaScript runtime. Found: {found}. "
            "Node.js 20 is unsupported; install Node.js 22+ or Deno 2.3+. "
            "In Colab, run !npm install -g deno, then add --js-runtimes deno to your command. "
            "No metadata or model downloads have been started."
        )
    args.js_runtimes = supported
    print(f"YouTube JavaScript runtime: {', '.join(descriptions)}", flush=True)


def download_process_error(log, returncode):
    """Expose yt-dlp's reason and distinguish dead uploads from setup failures."""
    try:
        with log.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 16384))
            tail = stream.read().decode("utf-8", errors="replace")
    except OSError:
        tail = ""
    tail = re.sub(r"\x1b\[[0-9;]*m", "", tail)
    errors = [line.strip() for line in tail.splitlines() if re.search(r"(?:^|: )error:", line, re.I)]
    reason = errors[-1] if errors else f"process exited with status {returncode}"
    message = f"Download failed ({reason}); details: {log}"
    lower = reason.lower().replace("’", "'")
    if "ffmpeg exited with code" in lower:
        # FFmpeg writes its own diagnostics without yt-dlp's ERROR: prefix.
        # An unsupported local option will fail for every source video.
        option_errors = re.findall(r"^Unrecognized option .+$", tail, re.M)
        if option_errors:
            return DownloadSetupError(
                f"Download failed ({option_errors[-1].strip()}); details: {log}. "
                "The installed FFmpeg does not support a download option. "
                "Use the updated train_downvid.py or update FFmpeg. "
                "Stopping instead of spending more clip attempts."
            )
    if "page needs to be reloaded" in lower:
        return DownloadSetupError(
            f"{message}. YouTube's player/session request failed; this does not establish that "
            "the source video is unavailable. Stopping instead of spending more clip attempts. "
            f"First update the downloader and EJS scripts in the environment running this command: "
            f"{sys.executable} -m pip install -U 'yt-dlp[default]'. "
            "Check for a supported JavaScript runtime (Node.js 22+ or Deno 2.3+), and remove "
            "custom --extractor-args while diagnosing. If the error persists, test the same URL "
            "with the same session on your local computer; refresh cookies with --login only "
            "if needed. A Colab IP block can persist even with valid cookies."
        )
    if any(value in lower for value in ("confirm you're not a bot", "confirm you are not a bot",
                                        "http error 429", "too many requests")):
        return DownloadSetupError(
            f"{message}. YouTube blocked or rate-limited access; stop retrying and check access "
            "in your browser. Save or refresh a browser session with --login --cookies FILE "
            "(Colab: upload exported YouTube cookies first), or --login --cookies-from-browser "
            "BROWSER on your local computer. Cookies may not resolve an IP-level block."
        )
    if (any(value in lower for value in ("no such option", "unsupported javascript runtime",
                                         "does not look like a netscape")) or
            ("cookie" in lower and any(value in lower for value in
                                       ("could not copy", "failed to decrypt", "could not find")))):
        return DownloadSetupError(
            f"{message}. Check downloader/runtime/cookie settings and update yt-dlp with "
            f"{sys.executable} -m pip install -U 'yt-dlp[default]'."
        )
    if any(value in lower for value in ("video unavailable", "video is unavailable", "private video",
                                        "video has been removed", "video has been deleted",
                                        "account associated with this video has been terminated")):
        return SourceUnavailableError(message)
    return RuntimeError(message)


def video_signature(path):
    stat = path.stat()
    return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]


def probe_video(path, expected_duration, max_frames=None):
    command = ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json"]
    if max_frames is not None:
        command.append("-count_frames")
    result = subprocess.run(
        [*command, str(path)],
        capture_output=True, text=True, check=True, timeout=60,
    )
    probe = json.loads(result.stdout)
    kinds = {stream.get("codec_type") for stream in probe.get("streams", [])}
    if not {"video", "audio"} <= kinds:
        raise ValueError("Downloaded clip must contain both video and audio for transcription")
    if max_frames is not None:
        stream = next(stream for stream in probe["streams"] if stream.get("codec_type") == "video")
        try:
            frames = int(stream.get("nb_read_frames", "0"))
        except (TypeError, ValueError) as exc:
            raise ValueError("Cannot count downloaded video frames") from exc
        if not 0 < frames <= max_frames:
            raise ValueError(f"Downloaded clip has {frames} frames; expected 1..{max_frames}")
        if frames == max_frames:
            # Reaching the cap legitimately shortens the requested section.
            # Use video duration, so an untrimmed audio tail still fails validation.
            video_duration = float(stream.get("duration", "nan"))
            if not math.isfinite(video_duration) or video_duration <= 0:
                raise ValueError("Cannot determine frame-limited video duration")
            expected_duration = min(expected_duration, video_duration) if expected_duration is not None else video_duration
    duration = float(probe["format"]["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Downloaded duration must be finite and positive")
    if (expected_duration is not None and
            abs(duration - expected_duration) > max(0.75, expected_duration * 0.1)):
        raise ValueError(f"Downloaded duration {duration:.3f}s differs from requested {expected_duration:.3f}s")
    return duration


def download_folder(clip, args):
    # Preserve existing full-clip caches; each frame limit gets its own download.
    settings = args.download_format if args.download_max_frames is None else {
        "format": args.download_format, "max_frames": args.download_max_frames,
    }
    if isinstance(clip, ArchiveClip):
        settings = {"dataset": "hdtf", "version": 1, "max_frames": args.download_max_frames}
    return args.work_dir / "clips" / clip.key / digest(settings)[:12]


def cached_download(clip, args):
    folder = download_folder(clip, args)
    video = folder / "video.mp4"
    saved = read_cache(folder / "download.json")
    if video.is_file() and saved.get("video_signature") == video_signature(video):
        return video
    return None


def run_download_process(command, *, stdout, timeout):
    """Stop yt-dlp and its FFmpeg children before removing temporary files."""
    with subprocess.Popen(command, stdout=stdout, stderr=subprocess.STDOUT,
                          start_new_session=os.name != "nt") as process:
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            try:
                if os.name == "nt":
                    # Killing only yt-dlp leaves FFmpeg downloading with the
                    # staging files open, also breaking temporary-file cleanup.
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=10, check=False)
                else:
                    os.killpg(process.pid, signal.SIGKILL)
            except (OSError, subprocess.SubprocessError):
                pass
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
            raise
        if code:
            raise subprocess.CalledProcessError(code, command)


def download_clip(clip, args):
    folder = download_folder(clip, args)
    folder.mkdir(parents=True, exist_ok=True)
    video = folder / "video.mp4"
    marker = folder / "download.json"
    cached = cached_download(clip, args)
    if cached is not None:
        print(f"  Reusing download: {cached}", flush=True)
        return cached

    log = folder / "download.log"
    with tempfile.TemporaryDirectory(prefix="download-", dir=folder) as staging:
        output = Path(staging) / "clip.mp4"
        command = [
            sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-playlist",
            "--no-continue", "--no-progress", "--socket-timeout", "30",
            "--retries", str(args.download_retries),
            "--fragment-retries", str(args.download_retries),
            "--format", args.download_format, "--merge-output-format", "mp4",
            "--remux-video", "mp4", "--force-keyframes-at-cuts",
            "--download-sections", f"*{clip.start:.9f}-{clip.end:.9f}",
            "--output", str(Path(staging) / "clip.%(ext)s"),
        ]
        if args.download_max_frames is not None:
            # Apply the limit while FFmpeg reads the remote section. Counting
            # decoded frames avoids estimating a duration from possibly VFR video.
            # trim signals video EOF; shortest limits audio to the video duration.
            # -vsync 0 requests timestamp passthrough on FFmpeg versions that
            # predate -fps_mode passthrough.
            command.extend([
                "--downloader", "ffmpeg", "--downloader-args",
                f"ffmpeg_o:-vf trim=end_frame={args.download_max_frames} -vsync 0 -shortest",
            ])
        for flag, value in [("--cookies", args.cookies),
                            ("--cookies-from-browser", args.cookies_from_browser),
                            ("--extractor-args", args.extractor_args)]:
            if value:
                command.extend([flag, str(value)])
        for runtime in download_runtimes(args):
            command.extend(["--js-runtimes", runtime])
        # yt-dlp's socket timeout does not apply to external FFmpeg downloads.
        # Limit network reads on EVERY input, including the audio stream.
        # FFmpeg 8.1 can inherit a large TCP read-ahead threshold and drain
        # megabytes of a remote MP4 before seeking. Force real HTTP seeks;
        # the socket timeout alone cannot stop a slow but active transfer.
        command.extend(["--downloader-args", "ffmpeg_i:-rw_timeout 30000000 -short_seek_size 1"])
        command.extend(["--", clip.url])
        try:
            with log.open("w", encoding="utf-8") as stream:
                run_download_process(command, stdout=stream, timeout=args.download_timeout)
            probe_video(output, clip.end - clip.start, max_frames=args.download_max_frames)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            if isinstance(exc, subprocess.CalledProcessError):
                raise download_process_error(log, exc.returncode) from exc
            elif isinstance(exc, subprocess.TimeoutExpired):
                reason = f"timed out after {exc.timeout}s"
            else:
                reason = str(exc)
            raise RuntimeError(f"Download failed ({reason}); details: {log}") from exc
        output.replace(video)
    write_json(marker, {"url": clip.url, "start": clip.start, "end": clip.end,
                        "max_frames": args.download_max_frames,
                        "video_signature": video_signature(video)})
    return video


class CharacterAlignmentTokenizer:
    """Expose the grouper's offset API using CESM's one-codepoint tokenization.

    IDs here are Unicode codepoints, solely for grouping. train.py builds its
    own character vocabulary from the completed manifest, as usual.
    """

    is_fast = True

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(ch) for ch in text],
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def convert_ids_to_tokens(self, token_id):
        return chr(token_id)


def alignment_tokenizer(args):
    import train

    resume_chars = None
    if args.resume:
        import torch

        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        state = checkpoint["tokenizer"]
        name = checkpoint["model_config"].get("pretrained_lm_name_or_path")
        if args.pretrained_lm and args.pretrained_lm != name:
            raise ValueError("--pretrained-lm does not match the resumed checkpoint")
        tokenizer = train.tokenizer_from_state_dict(state)
        if isinstance(tokenizer, train.CharTokenizer):
            if args.fine_tune_pretrained_lm:
                raise ValueError("--fine-tune-pretrained-lm requires a pretrained-LM checkpoint")
            resume_chars = state["itos"]
    elif args.pretrained_lm:
        tokenizer = train.HuggingFaceTokenizer.from_pretrained(
            args.pretrained_lm, local_files_only=args.pretrained_lm_local_files_only,
        )
    else:
        tokenizer = None
    if isinstance(tokenizer, train.HuggingFaceTokenizer):
        return tokenizer.tokenizer, "token", tokenizer.state_dict(), None
    return CharacterAlignmentTokenizer(), "character", {"type": "character"}, resume_chars


def load_grouper(path):
    spec = importlib.util.spec_from_file_location("_cesm_frame_grouper", path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves the class module through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def training_record(manifest, tokenizer, window_unit, video):
    """Reject dropped tokens and irreparable windows before invoking training."""
    import train

    text = manifest["text"]
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(value) for value in encoded["input_ids"]]
    if not text.strip() or not ids:
        raise ValueError("No speech tokens to train on")
    tokens = manifest["tokens"]
    if ids != [token["token_id"] for token in tokens]:
        raise ValueError("Frame grouper dropped tokens; alignment would not match the trainer")
    offsets = [tuple(value) for value in encoded["offset_mapping"]]
    if offsets != [(token["char_start"], token["char_end"]) for token in tokens]:
        raise ValueError("Frame grouper offsets do not match the training tokenizer")
    windows = manifest["groups"]
    if len(windows) != len(ids):
        raise ValueError("Expected one frame window per training token")
    cleaned, count = train.repair_quantized_empty_windows([tuple(win) for win in windows])
    previous = -1
    for start, end in cleaned:
        if not (previous < start <= end < manifest["frames_decoded"]):
            raise ValueError("Unusable frame windows: too few distinct frames for the transcript tokens")
        previous = end
    if count:
        print(f"  Trainer can repair {count} sub-frame token windows", flush=True)
    # Keep the original grouper windows: train.py applies exactly the validated
    # repair above. Explicit window_unit avoids equal token/character ambiguity.
    return {"video": str(video.resolve()), "text": text, "windows": windows,
            "window_unit": window_unit}


def prepare(args):
    import train

    grouper = load_grouper(args.group_frames_code)
    tokenizer, unit, token_state, resume_chars = alignment_tokenizer(args)
    config = {"version": 1, "tokenizer": token_state,
              "grouper_sha256": hashlib.sha256(args.group_frames_code.read_bytes()).hexdigest(),
              "whisper_model": args.whisper_model, "language": args.language,
              "device": args.whisper_device, "compute_type": args.whisper_compute_type,
              "image_ext": args.image_ext, "jpeg_quality": args.jpeg_quality}
    if args.download_max_frames is not None:
        config["download_max_frames"] = args.download_max_frames
    config_key = digest(config)[:24]
    source = args.hdtf_archive if args.dataset == "hdtf" else args.metadata
    report = {"dataset": args.dataset, "metadata": source, "requested": args.num_videos,
              "download_max_frames": args.download_max_frames,
              "manifest": str(args.manifest), "results": [], "failures": [],
              "unavailable_sources": {}, "skipped_unavailable_clips": 0}
    report_path = args.work_dir / "run_report.json"
    records, seen = [], set()
    attempts = 0
    whisper = None
    languages = {value.casefold() for value in args.dataset_language}
    if args.dataset == "hdtf" and "en" in languages:
        languages.add("english")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    partial = args.manifest.with_name(args.manifest.name + ".partial")
    print(f"Reading {args.dataset} source: {source}", flush=True)
    try:
        candidates = iter_hdtf(args) if args.dataset == "hdtf" else iter_metadata(args.metadata)
        with partial.open("w", encoding="utf-8") as jsonl, contextlib.closing(candidates) as rows:
            for index, row in enumerate(rows):
                if index < args.start_index:
                    continue
                if not isinstance(row, dict):
                    continue
                info = row.get("info") or {}
                if languages and (not isinstance(info, dict) or
                                  str(info.get("Language", "")).casefold() not in languages):
                    continue
                try:
                    clip = row["clip"] if args.dataset == "hdtf" else Clip.from_row(row)
                except (TypeError, ValueError, AttributeError) as exc:
                    report["failures"].append({"index": index, "error": f"Invalid metadata: {exc}"})
                    continue
                if clip.key in seen:
                    continue
                seen.add(clip.key)
                if (clip.source_key in report["unavailable_sources"] and
                        cached_download(clip, args) is None):
                    report["skipped_unavailable_clips"] += 1
                    continue
                attempts += 1
                description = (clip.member if isinstance(clip, ArchiveClip) else
                               f"{clip.url} [{clip.start:.3f}, {clip.end:.3f}]")
                print(f"[{len(records)}/{args.num_videos} usable; attempt {attempts}/{args.max_attempts}] "
                      + description, flush=True)
                try:
                    video = (download_hdtf_clip(clip, args, row["archive"], row["member"])
                             if args.dataset == "hdtf" else download_clip(clip, args))
                    output = args.work_dir / "frames" / clip.key / config_key
                    completed = output / "complete.json"
                    signature = video_signature(video)
                    cached = None
                    if not args.reprocess and completed.exists():
                        cached = read_cache(completed)
                    manifest = read_cache(output / "manifest.json")
                    if cached and cached.get("video_signature") == signature and manifest:
                        print(f"  Reusing alignment: {output}", flush=True)
                    else:
                        if whisper is None:
                            try:
                                whisper = grouper.WhisperModel(
                                    args.whisper_model, device=args.whisper_device,
                                    compute_type=args.whisper_compute_type,
                                )
                            except Exception as exc:
                                # A model/device setup error affects every clip;
                                # do not keep downloading replacement candidates.
                                raise WhisperSetupError(f"Cannot initialize Whisper: {exc}") from exc
                        # An interrupted/reprocessed run must not leave obsolete
                        # token folders beside a new alignment.
                        if output.exists():
                            shutil.rmtree(output)
                        print(f"  Transcribing and grouping frames: {video}", flush=True)
                        manifest = grouper.group_frames_by_token(
                            video, tokenizer, output, whisper_model=args.whisper_model,
                            language=args.language, device=args.whisper_device,
                            compute_type=args.whisper_compute_type, image_ext=args.image_ext,
                            jpeg_quality=args.jpeg_quality, whisper_instance=whisper,
                        )
                    record = training_record(manifest, tokenizer, unit, video)
                    grouper.write_jsonl([record], output / "data.jsonl")
                    write_json(completed, {"video_signature": signature, "config": config})
                    jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                    jsonl.flush()
                    records.append(record)
                    report["results"].append({"index": index, "id": row.get("id"),
                                               "clip": clip.__dict__, "output": str(output)})
                    print(f"  Ready: {len(record['windows'])} aligned tokens", flush=True)
                except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, subprocess.SubprocessError) as exc:
                    report["failures"].append({"index": index, "clip": clip.__dict__, "error": str(exc)})
                    if isinstance(exc, (WhisperSetupError, DownloadSetupError, ArchiveAccessError)):
                        raise
                    if isinstance(exc, SourceUnavailableError):
                        report["unavailable_sources"][clip.source_key] = str(exc)
                        print("  Source unavailable; skipping its remaining clips this run", flush=True)
                    print(f"  Skipping clip: {exc}", file=sys.stderr, flush=True)
                write_json(report_path, report)
                if len(records) == args.num_videos or attempts >= args.max_attempts:
                    break
        if len(records) != args.num_videos:
            raise RuntimeError(
                f"Only {len(records)}/{args.num_videos} usable clips after {attempts} attempts. "
                f"Training was not started. See {report_path}; partial data: {partial}. "
                "Increase --max-attempts or change the dataset/source/filter. Completed clips are cached."
            )
        if resume_chars is not None:
            vocabulary = train.CharTokenizer.build(record["text"] for record in records).itos
            if vocabulary != resume_chars:
                raise ValueError("--resume requires exactly the checkpoint's character vocabulary; "
                                 "the prepared transcripts differ. See the .partial manifest.")
        partial.replace(args.manifest)
        print(f"Prepared {len(records)} clips: {args.manifest}", flush=True)
    finally:
        report["usable"] = len(records)
        report["attempts"] = attempts
        write_json(report_path, report)


class WhisperSetupError(RuntimeError):
    pass


def training_command(args):
    """Serialize the actual trainer parser, keeping its defaults and all flags."""
    import train

    command = [sys.executable, str(ROOT / "train.py")]
    for action in train.build_argparser()._actions:
        if action.dest == "help":
            continue
        value = getattr(args, action.dest)
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                command.append(action.option_strings[0])
        elif value is not None:
            # The current trainer's other arguments are all single-value options.
            command.append(f"{action.option_strings[0]}={value}")
    return command


def validate_args(parser, args):
    if args.num_videos is None:
        parser.error("--num-videos/-N is required except with --login or --logout")
    args.work_dir = (args.work_dir or ROOT / "data" / args.dataset).expanduser().resolve()
    if args.dataset == "hdtf":
        if args.metadata != DEFAULT_METADATA:
            parser.error("--metadata is for TalkVid; use --hdtf-archive for HDTF")
        if any(language.casefold() not in ("english", "en") for language in args.dataset_language):
            parser.error("HDTF is an English dataset; --dataset-language must be English or en")
        if urlparse(args.hdtf_archive).scheme not in ("http", "https"):
            archive_path = Path(args.hdtf_archive).expanduser().resolve()
            if not archive_path.is_file():
                parser.error(f"HDTF archive not found: {archive_path}")
            args.hdtf_archive = str(archive_path)
    args.manifest = (Path(args.manifest).expanduser().resolve() if args.manifest
                     else args.work_dir / "data.jsonl")
    args.group_frames_code = args.group_frames_code.expanduser().resolve()
    if args.group_frames_code.is_dir():
        args.group_frames_code /= "group_video_frames_by_token.py"
    if not args.group_frames_code.is_file():
        parser.error(f"Frame grouping code not found: {args.group_frames_code}")
    if args.start_index < 0 or args.download_retries < 0:
        parser.error("--start-index and --download-retries must be >= 0")
    args.max_attempts = args.max_attempts or 10 * args.num_videos
    if args.max_attempts < args.num_videos:
        parser.error("--max-attempts must be >= --num-videos")
    if args.dataset == "talkvid":
        use_saved_login(args)
        if args.cookies and not args.cookies.expanduser().is_file():
            parser.error(f"Cookies file not found: {args.cookies}")
        if args.cookies:
            args.cookies = args.cookies.expanduser().resolve()
    if args.resume and not Path(args.resume).is_file():
        parser.error(f"Checkpoint not found: {args.resume}")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be > 0")
    if args.confidence_only and (not args.resume or args.confidence_epochs <= 0):
        parser.error("--confidence-only requires --resume and --confidence-epochs > 0")
    if args.epochs < 0 or args.confidence_epochs < 0:
        parser.error("--epochs and --confidence-epochs must be >= 0")
    if args.epochs == 0 and not args.resume and not args.prepare_only:
        parser.error("--epochs 0 requires --resume")
    if args.fine_tune_pretrained_lm and not (args.resume or args.pretrained_lm):
        parser.error("--fine-tune-pretrained-lm requires --pretrained-lm or a compatible --resume")
    dependencies = ["faster_whisper"]
    if args.dataset == "talkvid":
        dependencies.extend(["yt_dlp", "ijson"])
    missing = [name for name in dependencies
               if importlib.util.find_spec(name) is None]
    if missing:
        parser.error(f"Missing dependencies: {', '.join(missing)}. Install with: "
                     f"{sys.executable} -m pip install -r {ROOT / 'requirements-downvid.txt'}")
    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            parser.error(f"{binary} is required; install FFmpeg and make it available on PATH")
    if args.dataset == "talkvid":
        validate_download_runtimes(args)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_argparser()
    args = parser.parse_args(argv)
    if args.dataset != "talkvid" and (args.login or args.logout):
        parser.error("--login/--logout manage YouTube cookies for --dataset talkvid; HDTF needs no login")
    if args.login:
        login(args)
        return
    if args.logout:
        login_cookie_path(args).unlink(missing_ok=True)
        print("Removed saved YouTube cookies for this work directory. "
              "Your browser session and source cookie file are unchanged.", flush=True)
        return
    validate_args(parser, args)
    if args.prepare_only:
        prepare(args)
    else:
        # The preprocessing process exits before training allocates GPU memory,
        # releasing Whisper/CTranslate2 as well as tokenizer/checkpoint memory.
        subprocess.run([sys.executable, str(Path(__file__).resolve()), *argv, "--prepare-only"], check=True)
        command = training_command(args)
        print(f"Training: {shlex.join(command)}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; completed clips and alignments are cached.", file=sys.stderr)
        sys.exit(130)
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode if exc.returncode > 0 else 1)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
