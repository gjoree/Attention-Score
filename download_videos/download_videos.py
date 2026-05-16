#!/usr/bin/env python3
"""
Download multiple YouTube videos as MP4 files into an output folder.

Examples:
  python script.py --out videos "https://www.youtube.com/watch?v=abc" "https://youtu.be/xyz"
  python script.py --out videos                # reads urls.txt by default
  python script.py --out videos --urls-file my_urls.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download one or more YouTube URLs as MP4 files."
    )
    parser.add_argument(
        "urls",
        nargs="*",
        help="YouTube URLs to download.",
    )
    parser.add_argument(
        "--urls-file",
        type=Path,
        default=Path("urls.txt"),
        help="Path to a text file with one URL per line (default: urls.txt).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("videos"),
        help="Output folder (default: videos).",
    )
    return parser.parse_args()


def read_urls(urls_file: Path) -> list[str]:
    if not urls_file.exists():
        raise FileNotFoundError(f"URLs file not found: {urls_file}")

    urls: list[str] = []
    for raw_line in urls_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def collect_urls(args: argparse.Namespace) -> list[str]:
    urls = list(args.urls)
    if args.urls_file:
        urls.extend(read_urls(args.urls_file))

    # Keep order, remove duplicates
    seen: set[str] = set()
    unique_urls: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    return unique_urls


def main() -> int:
    args = parse_args()
    try:
        urls = collect_urls(args)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if not urls:
        print("No URLs found in the provided inputs.", file=sys.stderr)
        return 2

    try:
        import yt_dlp
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: yt-dlp\nInstall it with:\n  pip install yt-dlp"
        ) from exc

    args.out.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "outtmpl": str(args.out / "%(title)s [%(id)s].%(ext)s"),
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "restrictfilenames": True,
    }

    print(f"Downloading {len(urls)} video(s) to: {args.out.resolve()}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(urls)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
